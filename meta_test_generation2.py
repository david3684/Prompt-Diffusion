"""
    This file is for generating images for the FID calculation.
"""
import sys
sys.path.append("/data2/kietngt00/SD3Control")
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import yaml
import os
from datasets import load_dataset
import argparse
from src.sd3.pipeline_tools4 import pipeline_forward
from src.train.model4 import SD3Model
from src.dataset.coco2017val import TestDatamodule
from laion_meta_dataset import TASKS, ControlDataModule
from promptdiffusioncontrolnet_sd3 import SD3PromptDiffusionModel
from promptdiffusioncontrolnetpipeline_sd3 import SD3PromptDiffusionPipeLine
from tqdm import tqdm
from einops import rearrange
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np


def visualize_generation(gt, cond, image, sp_cond, sp_image, prompt):
    n_col = 1 + len(sp_image)
    n_row = 3
    plt.figure(figsize=(2*n_col, 2*n_row))
    plt.suptitle(f"{prompt}")
    plt.subplot(n_row, n_col, 1)
    plt.imshow(cond)
    plt.axis('off')
    plt.title("Query")

    for i, c in enumerate(sp_cond):
        plt.subplot(n_row, n_col, i+2)
        plt.imshow(c)
        plt.axis('off')
        title = f"Support {i}"
        plt.title(title)

    plt.subplot(n_row, n_col, n_col + 1)
    plt.imshow(image)
    plt.axis('off')

    for i, img in enumerate(sp_image):
        plt.subplot(n_row, n_col, i+2+n_col)
        plt.imshow(img)
        plt.axis('off')
    
    plt.subplot(n_row, n_col, n_col * 2 + 1)
    plt.imshow(gt)
    plt.axis('off')
    plt.title("GT")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    pil_image = Image.open(buf).convert('RGB')
    buf.close()
    return pil_image


def tensor_to_image(batch):
    batch['task_indices'] = batch['task_indices'].squeeze(0)

    batch['images'] = batch['images'].squeeze(0)
    batch['images'] = torch.chunk(batch['images'], 2, dim=0)[0]
    images = []
    for i in range(batch['images'].shape[0]):
        img = batch['images'][i]
        img = rearrange(img, 'C H W -> H W C')
        img = (img * 255).byte().numpy()
        img = Image.fromarray(img)
        images.append(img)
    batch['images'] = images

    batch['conditions'] = batch['conditions'].squeeze(0)
    batch['conditions'] = torch.chunk(batch['conditions'], 2, dim=0)[0]
    conditions = []
    for t in range(batch['conditions'].shape[0]):
        task_conds = []
        for i in range(batch['conditions'].shape[1]):
            cond = batch['conditions'][t][i]
            cond = rearrange(cond, 'C H W -> H W C')
            cond = cond.byte().numpy()
            cond = Image.fromarray(cond)
            task_conds.append(cond)
        conditions.append(task_conds)
    batch['conditions'] = conditions

    prompts = batch['prompts']
    batch['prompts'] = [prompts[i][0] for i in range(len(batch['images']))] # an element in prompts is a tuple of len 1

    return batch


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    training_config = config["train"]
    data_config = config["data"]

    # Model
    model = SD3Model(
        sd3_pipe_id="stabilityai/stable-diffusion-3.5-medium",
        device=f"cuda:{args.gpu}",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    ckpt_path = args.ckpt_path
    # state_dict = torch.load(ckpt_path, map_location="cpu")
    # model.matching_modules.load_state_dict(state_dict["state_dict"]["matching_modules"])
    model.load_bias_params(ckpt_path)
    model = model.eval()
    model = model.to(f"cuda:{args.gpu}")

    # Data
    test_tasks = data_config["test_tasks"] if not args.task else [args.task]
    datamodule = TestDatamodule(
        path="/data2/kietngt00/coco2017/val2017",
        tasks=test_tasks,
        res=512,
        batch_size=args.batch_size,
        num_workers=0,
    )
    test_loader = datamodule.test_dataloader()

    datamodule = ControlDataModule(path=data_config["path"],
                                human_path=data_config["human_path"],
                                train_tasks=data_config["train_tasks"],
                                test_tasks=data_config["test_tasks"],
                                tasks_per_batch=data_config["tasks_per_batch"],
                                splits=data_config["splits"],
                                shots=data_config["shots"],
                                batch_size=data_config["batch_size"],
                                num_workers=data_config["num_workers"],)
    tuning_dl = datamodule.tuning_dataloader(test_tasks, 15, 15) # TODO: check num sp and shots. Goal: get all support pairs in 1 batch
    supports = next(iter(tuning_dl))
    supports = tensor_to_image(supports)

    # Save path
    exp_name = ckpt_path.split("/")[2]
    tuning_ckpt = ckpt_path.split("/")[-1].split(".")[0]
    output_dir = f"./tuning_generation/{exp_name}/{tuning_ckpt}"
    print("save at", output_dir)
    for task in test_tasks:
        os.makedirs(output_dir + f"/{task}", exist_ok=True)
        if args.compute_fid:
            os.makedirs(f"{output_dir}/{task}/fid", exist_ok=True)

    # Generation
    for idx, batch in tqdm(enumerate(test_loader)):
        if not args.compute_fid and idx == 10:
            break
        gt_images = batch['images'] # List: B * T PIL images
        q_cond = batch['q_cond'] # List: B * T PIL images
        prompts = batch['prompts'] # List: B * T
        task_indices = batch['task_indices'] # List: B * T
        filenames = batch['filenames'] # List: B * T
        T = len(test_tasks)
        B = len(batch['q_cond']) // T
        S = data_config['shots']

        assert torch.equal(task_indices[:T], supports['task_indices']), "Task indices do not match"
        sp_image = supports['images'] # S images
        sp_cond = supports['conditions'] # T [S images]

        if args.check_support:
            if idx >= len(supports['images']) // S:
                break

            start = S * idx
            end = S * (idx + 1)
            q_cond = [[cond[i] for i in range(start, end)] for cond in sp_cond] # List: T [S images]
            prompts = [supports['prompts'][i] for i in range(start, end)]
            if idx < len(supports['images']) // S - 1:
                next_end = S * (idx + 2)
                sp_image = [sp_image[i] for i in range(end, next_end)] # List: S images
                sp_cond = [[cond[i] for i in range(end, next_end)] for cond in sp_cond] # List: T [S images]
            elif idx == len(supports['images']) // S - 1:
                sp_image = [sp_image[i] for i in range(S)] # List: S images
                sp_cond = [[cond[i] for i in range(S)] for cond in sp_cond] # List: T [S images]

            q_cond = [item for sub_q_cond in q_cond for item in sub_q_cond] # List: [q_task1, q_task1, q_task2, q_task2]
            sp_cond = [item for item in sp_cond for _ in range(S)] # List: [sp_task1, sp_task1, sp_task2, sp_task2]
            sp_image = [sp_image] * (S * T)
            prompts = prompts * T # List: [p_t1, p_t1, p_t2, p_t2]
            task_indices = supports['task_indices']
            task_indices = task_indices.unsqueeze(1).repeat(1, S ) # T S
            task_indices = rearrange(task_indices, 'T S -> (T S)')# [t1 t1 t2 t2]
            
        else:
            # Randomly choose S support pairs
            indices = np.random.choice(len(sp_image), S, replace=False)
            sp_image = [[sp_image[i] for i in indices]] * (B * T)
            sp_cond = [[cond[i] for i in indices] for cond in sp_cond] * B


        with torch.no_grad():
            images, _, _, _ = pipeline_forward( # (B T) or (S T) if check_support
                model,
                width=512,
                height=512,
                prompt=prompts,
                negative_prompt="lowres, low quality, worst quality",
                num_inference_steps=24, 
                guidance_scale=5.0,
                return_dict=False,
                q_cond=q_cond,
                sp_cond=sp_cond,
                sp_image=sp_image,
                task_indices=task_indices
            )

        if args.check_support:
            for i, task in enumerate(test_tasks):
                plt.figure(figsize=(2*S, 2*3))
                for j in range(S):
                    plt.subplot(3, S, j+1)
                    plt.imshow(q_cond[i*S + j])
                    plt.axis('off')
                    plt.subplot(3, S, j+S + 1)
                    plt.imshow(images[i*S + j])
                    plt.axis('off')
                    plt.subplot(3, S, j+2*S + 1)
                    plt.imshow((supports['images'][idx*S + j]))
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{task}_support_{idx}.png")
                plt.close()
                # if idx == 1:
                #     plt.figure(figsize=(10*2, 10))
                #     img1 = Image.open(f"{output_dir}/{task}_support_0.png")
                #     img2 = Image.open(f"{output_dir}/{task}_support_1.png")
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(img1)
                #     plt.axis('off')
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(img2)
                #     plt.axis('off')
                #     plt.tight_layout()
                #     plt.savefig(f"{output_dir}/{task}_support.png")
                #     plt.close()
        else:
            for i in range(len(images)):
                img = images[i]
                task = test_tasks[i % T]
                if args.compute_fid:
                        img.save(f"{output_dir}/{task}/fid/{filenames[i]}")
                else:
                    plot = visualize_generation(gt_images[i], q_cond[i], img, sp_cond[i], sp_image[i], prompts[i])
                    plot.save(f"{output_dir}/{task}/{filenames[i]}")
        


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--ckpt_path', type=str, required=True)
    args.add_argument('--gpu', type=int, default=0)
    args.add_argument('--compute_fid', action='store_true')
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--check_support', action='store_true')
    args.add_argument('--task', type=str, required=False)
    args = args.parse_args()

    main(args)







