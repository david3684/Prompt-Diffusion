#!/usr/bin/env python
# coding=utf-8

import os
import sys
import argparse
import torch
import yaml
import numpy as np
from tqdm import tqdm
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from torch.utils.data import DataLoader

from promptdiffusioncontrolnet_sd3 import SD3PromptDiffusionModel
from promptdiffusioncontrolnetpipeline_sd3 import SD3PromptDiffusionPipeLine
from laion_meta_dataset import TASKS, ControlDataModule
from coco2017val import TestDatamodule


def taskidx_to_taskname(task_idx):
    for k, v in TASKS.items():
        if v == task_idx:
            return k
    return "unknown_task"


def tensor_to_image(batch):
    batch['task_indices'] = batch['task_indices'].squeeze(0)

    print("batch['images'] shape:", batch['images'].shape) # (B, 15*T, 3, 512, 512)
    batch['images'] = batch['images'].squeeze(0)
    batch['images'] = torch.chunk(batch['images'], 2, dim=0)[0] # use only first 15 images (same)
    print("batch['images'] shape:", batch['images'].shape) # (15*1, 3, 512, 512)

    images = []
    for i in range(batch['images'].shape[0]):
        img = batch['images'][i]
        img = rearrange(img, 'C H W -> H W C')
        img = (img * 255).byte().numpy()
        img = Image.fromarray(img)
        images.append(img)
    batch['images'] = images
    print(np.shape(batch['conditions'])) #(B, T, 15*T, 3, 512, 512)
    batch['conditions'] = batch['conditions'].squeeze(0) #
    print(np.shape(batch['conditions'])) #(B, T, 15*T, 3, 512, 512)
    batch['conditions'] = torch.chunk(batch['conditions'], 2, dim=0)[0]

    conditions = []
    for t in range(batch['conditions'].shape[0]):
        task_conds = []
        for i in range(batch['conditions'].shape[1]):
            cond = batch['conditions'][t][i]
            cond = rearrange(cond, 'C H W -> H W C')
            cond = (cond*255).byte().numpy()
            cond = Image.fromarray(cond)
            task_conds.append(cond)
        conditions.append(task_conds)
    print("Converted conditions structure:", [len(task_conds) for task_conds in conditions])
    batch['conditions'] = conditions

    prompts = batch['prompts']
    batch['prompts'] = [prompts[i][0] for i in range(len(batch['images']))]  # an element in prompts is a tuple of len 1

    return batch


def visualize_generation(gt, cond, image, sp_cond, sp_image, prompt, task_name=None):
    n_col = 1 + len(sp_image)
    n_row = 3
    
    plt.figure(figsize=(2*n_col, 2*n_row))
    title = f"{prompt}"
    if task_name:
        title = f"Task: {task_name} - {title}"
    plt.suptitle(title)
    
    plt.subplot(n_row, n_col, 1)
    plt.imshow(cond)  
    plt.axis('off')
    plt.title("Query Condition")
    
    for i, c in enumerate(sp_cond):
        plt.subplot(n_row, n_col, i+2)
        plt.imshow(c)  
        plt.axis('off')
        plt.title(f"Support {i+1}")
    
    plt.subplot(n_row, n_col, n_col+1)
    image_np = np.array(image).astype(np.float32) / 255.0
    plt.imshow(np.clip(image_np, 0, 1))
    plt.axis('off')
    plt.title("Generated")
    
    for i, img in enumerate(sp_image):
        plt.subplot(n_row, n_col, n_col+i+2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Support {i+1}")
    
    plt.subplot(n_row, n_col, 2*n_col+1)
    plt.imshow(gt)
    plt.axis('off')
    plt.title("Ground Truth")
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=150)
    plt.close()
    buf.seek(0)
    pil_image = Image.open(buf).convert('RGB')
    buf.close()
    
    return pil_image


def main():
    parser = argparse.ArgumentParser(description="Prompt Diffusion 이미지 생성기")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--output_dir", type=str, default="./outputs/generated")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=24)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--max_images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute_fid", action="store_true")
    parser.add_argument("--coco_path", type=str, default="/data2/david3684/ufg_diff/datasets/coco2017/val2017")
    parser.add_argument("--black_support", action="store_true")
    args = parser.parse_args()
    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config.get("data", {})

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.checkpoint_path}...")
    controlnet = SD3PromptDiffusionModel.from_pretrained(
        args.checkpoint_path, torch_dtype=torch.float16
    )
    
    print(f"Loading base model from {args.base_model}...")
    pipe = SD3PromptDiffusionPipeLine.from_pretrained(
        args.base_model, controlnet=controlnet
    )
    pipe.to(device, torch.float16)
    
    # 태스크 설정
    train_tasks = data_config.get("train_tasks", ["canny", "depth", "hed", "normal"])
    test_tasks = data_config.get("test_tasks", ['pose', 'densepose'])
    
    if not test_tasks:
        test_tasks = train_tasks
    
    print(f"Using test tasks: {test_tasks}")
    
    print("Loading COCO test dataset...")
    coco_datamodule = TestDatamodule(
        path=args.coco_path,
        tasks=test_tasks,
        res=512,
        batch_size=args.batch_size,
        num_workers=data_config.get("num_workers", 0),
    )
    test_loader = coco_datamodule.test_dataloader()
    
    print("Loading LAION support dataset...")
    laion_datamodule = ControlDataModule(
        path=data_config.get("path", "/data2/david3684/ufg_diff/datasets/laion_data/laion_nonhuman"),
        human_path=data_config.get("human_path", "/data2/david3684/ufg_diff/datasets/laion_data/laion_human"),
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        tasks_per_batch=data_config.get("tasks_per_batch", 1),
        splits=(0.9, 0.1),
        res=512,
        shots=data_config.get("shots", 1),
        batch_size=data_config.get("batch_size", 1),
        num_workers=data_config.get("num_workers", 2),
        total_samples=data_config.get("total_samples", 10000),
    )
    laion_datamodule.setup()
    support_shots = data_config.get("shots", 1)
    tuning_dl = laion_datamodule.tuning_dataloader(test_tasks, num_supports=15, shots=15)
    supports = next(iter(tuning_dl))
    supports = tensor_to_image(supports) 
    
    
    model_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    ckpt_name = os.path.basename(args.checkpoint_path)
    if "." in ckpt_name:
        ckpt_name = ckpt_name.split(".")[0]
    
    output_dir = os.path.join(args.output_dir, model_name, ckpt_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for task in test_tasks:
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        if args.compute_fid:
            os.makedirs(os.path.join(task_dir, "fid"), exist_ok=True)
    
    print("Starting image generation...")
    total_generated = 0
    
    for idx, batch in enumerate(tqdm(test_loader, desc="Generating images")):
        if total_generated >= args.max_images:
            break
        

        gt_images = batch['images'] # List: B * T PIL 이미지
        q_cond = batch['q_cond'] # List: B * T PIL 이미지 
        prompts = batch['prompts'] # List: B * T
        task_indices = batch['task_indices'] # List: B * T
        filenames = batch['filenames'] # List: B * T
        print(filenames[0])
        T = len(test_tasks)
        B = len(batch['q_cond']) // T
        S = support_shots

        sp_image = supports['images']  # S 이미지 15 * H * W * C
        sp_cond = supports['conditions']  # T [S 이미지] # B, T*15, H, W, C

        for i in range(len(q_cond)):
            if total_generated >= args.max_images:
                break
                
            # 현재 쿼리 이미지 및 태스크 정보
            curr_q_cond = q_cond[i]
            curr_prompt = prompts[i]
            curr_task_idx = task_indices[i]
            curr_task_name = taskidx_to_taskname(curr_task_idx)
            curr_filename = filenames[i]
            curr_gt_image = gt_images[i]
            
            task_index_in_supports = -1
            for t, t_idx in enumerate(supports['task_indices']):
                if t_idx.item() == curr_task_idx:
                    task_index_in_supports = t
                    break
                    
            if task_index_in_supports == -1:
                print(f"Warning: Task {curr_task_name} (index {curr_task_idx}) not found in supports. Skipping.")
                continue
                
            sp_indices = np.random.choice(len(sp_image), S, replace=False)
            curr_sp_image = [sp_image[j] for j in sp_indices]

            # curr_sp_cond = [sp_cond[0][j+15*task_index_in_supports] for j in sp_indices]

            curr_sp_cond = [sp_cond[task_index_in_supports][j] for j in sp_indices]
            
            query_control = torch.from_numpy(np.array(curr_q_cond)).permute(2, 0, 1).float().to(device).unsqueeze(0) / 255.0
            support_images = []
            support_controls = []
            
            for j in range(len(curr_sp_image)):
                if args.black_support:
                    sp_img = torch.zeros(1, 3, 512, 512, device=device, dtype=torch.float16)
                    sp_ctrl = torch.zeros(1, 3, 512, 512, device=device, dtype=torch.float16)
                else:
                    sp_img = torch.from_numpy(np.array(curr_sp_image[j])).permute(2, 0, 1).float().to(device).unsqueeze(0) / 255.0
                    sp_ctrl = torch.from_numpy(np.array(curr_sp_cond[j])).permute(2, 0, 1).float().to(device).unsqueeze(0) / 255.0
                support_images.append(sp_img)
                support_controls.append(sp_ctrl)
            
            with torch.no_grad():
                for sp_idx in range(len(support_images)):
                    generated = pipe(
                        prompt=curr_prompt,
                        height=512,
                        width=512,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        negative_prompt="lowres, low quality, worst quality, blurry",
                        control_image=query_control,
                        control_image_pair=[support_controls[sp_idx], support_images[sp_idx]],
                        controlnet_conditioning_scale=args.conditioning_scale,
                        num_images_per_prompt=1,
                        generator=torch.Generator(device=device).manual_seed(args.seed + total_generated),
                    )
                    
                    # 생성된 이미지
                    generated_image = generated.images[0]
                    
                    if args.compute_fid:
                        fid_save_path = os.path.join(
                            output_dir, curr_task_name, "fid", f"{curr_filename}"
                        )
                        generated_image.save(fid_save_path)
                    else:
                        vis_dir = os.path.join(output_dir, curr_task_name, "visualizations")
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        raw_dir = os.path.join(output_dir, curr_task_name, "raw_images")
                        os.makedirs(raw_dir, exist_ok=True)
                        
                        
                        filename_suffix = "_black" if args.black_support else ""
                        raw_save_path = os.path.join(
                            raw_dir, f"{os.path.splitext(curr_filename)[0]}{filename_suffix}_sp{sp_idx}.png"
                        )
                        generated_image.save(raw_save_path)
                        
                        query_control_np = query_control[0].permute(1, 2, 0).cpu().numpy()
                        
                        if args.black_support:
                            support_control_np = np.zeros((512, 512, 3))
                            support_gt_np = np.zeros((512, 512, 3))
                        else:
                            support_control_np = support_controls[sp_idx][0].permute(1, 2, 0).cpu().numpy()
                            support_gt_np = support_images[sp_idx][0].permute(1, 2, 0).cpu().numpy()
                        
                        gt_img_np = np.array(curr_gt_image).astype(np.float32) / 255.0
                        
                        filename_suffix = "black_sp" if args.black_support else "sp"
                        vis_save_path = os.path.join(
                            vis_dir, f"{os.path.splitext(curr_filename)[0]}_{filename_suffix}{sp_idx}.jpg"
                        )
                        print(f"Saving visualization to {vis_save_path}")   
                        
                        plot = visualize_generation(
                            gt=gt_img_np,
                            cond=query_control_np,
                            image=generated_image,
                            sp_cond=[support_control_np],
                            sp_image=[support_gt_np],
                            prompt=curr_prompt,
                            task_name=curr_task_name
                        )
                        plot.save(vis_save_path)
                    
                    total_generated += 1
    
    print(f"Total Image Generated: {total_generated}")


if __name__ == "__main__":
    main()