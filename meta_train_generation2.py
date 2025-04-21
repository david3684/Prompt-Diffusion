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
from src.dataset.laion_meta_dataset import TASKS, ControlDataModule
from tqdm import tqdm
from einops import rearrange
from matplotlib import pyplot as plt
from io import BytesIO
import numpy as np
from PIL import Image

def taskidx_to_taskname(task_idx):
    for k, v in TASKS.items():
        if v == task_idx:
            return k


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
    plt.savefig(buf, format='jpg')
    plt.close()
    buf.seek(0)

    pil_image = Image.open(buf).convert('RGB')
    buf.close()
    return pil_image



def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    training_config = config["train"]
    data_config = config["data"]

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


    bs = args.batch_size
    datamodule = ControlDataModule(path=data_config["path"],
                                   human_path=data_config["human_path"],
                                   train_tasks=data_config["train_tasks"],
                                   test_tasks=data_config["train_tasks"],
                                   tasks_per_batch=data_config["tasks_per_batch"],
                                   splits=data_config["splits"],
                                   shots=data_config["shots"],
                                   batch_size=data_config["batch_size"],
                                   num_workers=data_config["num_workers"],)
    datamodule.setup()
    if args.compute_fid:
        test_ds = datamodule.val_ds
    else:
        indices = list(range(10)) + list(range(5000,5010))
        test_ds = torch.utils.data.Subset(datamodule.val_ds, indices)
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=0, shuffle=False)

    exp_name = ckpt_path.split("/")[2]
    epoch = ckpt_path.split("/")[-1].split(".")[0]
    output_dir = f"./generation/{exp_name}/{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    for t in data_config["train_tasks"]:
        os.makedirs(f"{output_dir}/{t}", exist_ok=True)

    for idx, batch in tqdm(enumerate(test_loader)):
        if idx * bs * data_config['shots'] > 5000:
            break
        imgs = batch["images"]              # [B 2*shot C H W]
        conditions = batch["conditions"]    # [B T 2*shot C H W]
        prompts = batch["prompts"]          # shots x [B]
        task_indices = batch["task_indices"] # [B T]
        log_task_indices = task_indices.clone().unsqueeze(2).repeat(1, 1, data_config["shots"])
        log_task_indices = rearrange(log_task_indices, 'B T S -> (B T S)')
        task_indices = rearrange(task_indices, 'B T -> (B T)')

        tasks = []
        for t_idx in task_indices:
            tasks.append(taskidx_to_taskname(t_idx))

        B, T, S = conditions.shape[:3]
        S //= 2

        q_cond, sp_cond = torch.chunk(conditions, 2, dim=2)
        gt_image, sp_image = torch.chunk(imgs, 2, dim=1)

        prompts2 = []
        for i in range(B):
            temp = []
            for j in range(S):
                temp.append(prompts[j][i])
            temp = temp * T
            prompts2 += temp
        prompts = prompts2 # (B T S)

        with torch.no_grad():
            images, _, _, _ = pipeline_forward( # (B T S)
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

        if args.compute_fid:
            for i in range(len(images)):
                img = images[i]
                img.save(f"{output_dir}/{tasks[i//S]}/{idx}_{i}.jpg")
        else:
            gt_image = gt_image.unsqueeze(1).repeat(1, T, 1, 1, 1, 1)
            gt_image = rearrange(gt_image, 'B T S C H W -> (B T S) H W C')
            q_cond = rearrange(q_cond, 'B T S C H W -> (B T S) H W C') / 255.
            sp_cond = rearrange(sp_cond, 'B T S C H W -> (B T) S H W C') / 255.
            sp_image = sp_image.unsqueeze(1).repeat(1, T, 1, 1, 1, 1)
            sp_image = rearrange(sp_image, 'B T S C H W -> (B T) S H W C')
            outdir = f"{output_dir}/{idx}"
            os.makedirs(outdir, exist_ok=True)
            for i in range(len(images)):
                img = images[i]
                cond = q_cond[i]
                sp_c = sp_cond[i//S]
                sp_i = sp_image[i//S]
                plot = visualize_generation(gt_image[i], cond, img, sp_c, sp_i, prompts[i])
                plot.save(f"{outdir}/{i}_{tasks[i//S]}.jpg")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--ckpt_path', type=str, required=True)
    args.add_argument('--gpu', type=int, default=0)
    args.add_argument('--compute_fid', action='store_true')
    args.add_argument('--batch_size', type=int, default=1)
    args = args.parse_args()

    main(args)