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
from pipeline_prompt_diffusion import PromptDiffusionPipeline
from promptdiffusioncontrolnet import PromptDiffusionControlNetModel
from promptdiffusioncontrolnet_sd3 import SD3PromptDiffusionModel
from promptdiffusioncontrolnetpipeline_sd3 import SD3PromptDiffusionPipeLine
from laion_meta_dataset import TASKS, ControlDataModule
from coco2017val import TestDatamodule


def taskidx_to_taskname(task_idx):
    for k, v in TASKS.items():
        if v == task_idx:
            return k
    return "unknown_task"


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
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="./outputs/generated")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gen_batch_size", type=int, default=1)
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


    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.checkpoint_path}...")
    controlnet = PromptDiffusionControlNetModel.from_pretrained(
        args.checkpoint_path
    )
    
    print(f"Loading base model from {args.base_model}...")
    pipe = PromptDiffusionPipeline.from_pretrained(
        args.base_model, controlnet=controlnet
    )
    pipe.to(device, torch.float16)

    tasks = ['canny']
    
    print(f"Using test tasks: {tasks}")
    
    print("Loading COCO test dataset...")
    coco_datamodule = TestDatamodule(
        path=args.coco_path,
        tasks=tasks,
        res=512,
        batch_size=args.gen_batch_size,
        num_workers=4,
    )
    test_loader = coco_datamodule.test_dataloader()
    
    print("Loading LAION support dataset...")
    laion_datamodule = ControlDataModule(
        path="/data2/david3684/ufg_diff/datasets/laion_data/laion_nonhuman",
        human_path="/data2/david3684/ufg_diff/datasets/laion_data/laion_human",
        train_tasks=tasks,
        test_tasks=tasks,
        tasks_per_batch=1,
        splits=(1.0,0.0),
        res=512,
        shots=1,
        batch_size=1,
        num_workers=4,
        total_samples=150000,
    )
    laion_datamodule.setup()
    tuning_dl = laion_datamodule.tuning_dataloader(args.task, num_supports=15, shots=1)
    supports = next(iter(tuning_dl))
    
    
    model_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    ckpt_name = os.path.basename(args.checkpoint_path)
    if "." in ckpt_name:
        ckpt_name = ckpt_name.split(".")[0]
    
    output_dir = os.path.join(args.output_dir, model_name, ckpt_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for task in tasks:
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        if args.compute_fid:
            os.makedirs(os.path.join(task_dir, "fid"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "raw_images"), exist_ok=True)
    
    print("Starting image generation...")
    total_generated = 0
    batch_size = args.gen_batch_size
    for idx, batch in enumerate(tqdm(test_loader, desc="Generating images")):
        if total_generated >= args.max_images:
            break
        
        # Test dataset batch
        gt_images = batch['images']  # List: B PIL 
        q_cond = batch['q_cond']     # List: B 
        prompts = batch['prompts']   # List: B 
        task_indices = batch['task_indices']  # Tensor: [B * T]
        filenames = batch['filenames']
    
        
        supports = next(iter(tuning_dl))
        
        sp_images = supports['images'] # B , 2 , C , H , W
        sp_img = sp_images[:, 1, :, :, :]
        sp_conds = supports['conditions'] # B , T , 2 , C , H , W 
        sp_cond = sp_conds[:, :, 1, :, :, :]
        B, T, C, H, W = sp_cond.shape
        sp_cond = sp_cond.view(B * T, C, H, W) # T= 1 
        
        sp_img = sp_img.repeat(batch_size, 1, 1, 1)  # Shape: (batch_size, C, H, W)
        sp_cond = sp_cond.repeat(batch_size, 1, 1, 1)

        # query control from test dataloader
        # query prompt from test dataloader 
        with torch.no_grad():
            generated = pipe(
                prompt=prompts,
                image=q_cond,
                image_pair=[sp_cond, sp_img],
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                negative_prompt="lowres, low quality, worst quality",
                controlnet_conditioning_scale=args.conditioning_scale,
                generator=torch.Generator(device=device).manual_seed(args.seed + total_generated),
            )
            
            # 생성된 이미지
            generated_image = generated.images[0]

        for i in range(len(generated.images)):
            # Get task name
            curr_task_idx = task_indices[i].item()
            curr_task_name = taskidx_to_taskname(curr_task_idx)
            task_dir = os.path.join(output_dir, curr_task_name)
            
            # Convert tensors to Numpy for visualization
            gt_img_np = gt_images[i].cpu().float().permute(1, 2, 0).numpy()  # Shape: (H, W, C)
            gt_img_np = (gt_img_np * 255.0).astype(np.uint8)  # [0, 1] -> [0, 255]
            query_control_np = q_cond[i].cpu().float().permute(1, 2, 0).numpy()  # Shape: (H, W, C)
            query_control_np = (query_control_np * 255.0).astype(np.uint8)  # [0, 1] -> [0, 255]
            generated_image = generated.images[i]  # PIL (pipe 출력)
            
            # Convert support tensors to Numpy for visualization
            support_control_np = sp_cond[0].cpu().float().permute(1, 2, 0).numpy()  # Shape: (H, W, C)
            support_control_np = (support_control_np * 255.0).astype(np.uint8)  # [0, 1] -> [0, 255]
            support_gt_np = sp_img[0].cpu().float().permute(1, 2, 0).numpy()  # Shape: (H, W, C)
            support_gt_np = (support_gt_np * 255.0).astype(np.uint8)  # [0, 1] -> [0, 255]
            curr_prompt = prompts[i]
            
            # Save visualization
            vis_save_path = os.path.join(task_dir, "visualizations", f"{filenames[i]}_vis.jpg")
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
                    
            raw_save_path = os.path.join(task_dir, "raw_images", f"{filenames[i]}_gen.jpg")
            generated_image.save(raw_save_path)
            
            # Save for FID if required
            if args.compute_fid:
                fid_save_path = os.path.join(task_dir, "fid", f"{filenames[i]}_fid.jpg")
                generated_image.save(fid_save_path)
            
            total_generated += 1
            if total_generated >= args.max_images:
                break
    
    print(f"Total Images Generated: {total_generated}")


if __name__ == "__main__":
    main()