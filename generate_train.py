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
    """태스크 인덱스를 태스크 이름으로 변환"""
    for k, v in TASKS.items():
        if v == task_idx:
            return k
    return "unknown_task"


def visualize_generation(gt, cond, image, sp_cond, sp_image, prompt, task_name=None):
    """
    결과 시각화 함수 (모든 이미지가 0~1 범위로 정규화된 것으로 가정)
    
    Args:
        gt: 정답 이미지 (0~1 범위, numpy 배열)
        cond: 쿼리 조건 이미지 (0~1 범위, numpy 배열)
        image: 생성된 이미지 (PIL 이미지)
        sp_cond: 서포트 조건 이미지 리스트 (0~1 범위, numpy 배열)
        sp_image: 서포트 이미지 리스트 (0~1 범위, numpy 배열)
        prompt: 생성에 사용된 프롬프트
        task_name: 태스크 이름 (canny, depth 등)
    """
    n_col = 1 + len(sp_image)
    n_row = 3
    
    plt.figure(figsize=(2*n_col, 2*n_row))
    title = f"{prompt}"
    if task_name:
        title = f"Task: {task_name} - {title}"
    plt.suptitle(title)
    
    # 1. 조건 이미지들 표시 - 이미 0~1 범위로 정규화되어 있음
    plt.subplot(n_row, n_col, 1)
    plt.imshow(cond)  # 0~1 범위 유지
    plt.axis('off')
    plt.title("Query Condition")
    
    for i, c in enumerate(sp_cond):
        plt.subplot(n_row, n_col, i+2)
        plt.imshow(c)  # 0~1 범위 유지
        plt.axis('off')
        plt.title(f"Support {i+1}")
    
    # 2. 생성된 이미지와 서포트 이미지 표시
    plt.subplot(n_row, n_col, n_col+1)
    image_np = np.array(image).astype(np.float32) / 255.0
    plt.imshow(np.clip(image_np, 0, 1))
    plt.axis('off')
    plt.title("Generated")
    
    for i, img in enumerate(sp_image):
        plt.subplot(n_row, n_col, n_col+i+2)
        # 서포트 이미지는 이미 0~1 범위로 변환됨
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Support {i+1}")
    
    # 3. 정답 이미지 표시
    plt.subplot(n_row, n_col, 2*n_col+1)
    # 정답 이미지는 이미 0~1 범위로 변환됨
    plt.imshow(gt)
    plt.axis('off')
    plt.title("Ground Truth")
    
    plt.tight_layout()
    
    # 이미지로 변환
    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=150)
    plt.close()
    buf.seek(0)
    pil_image = Image.open(buf).convert('RGB')
    buf.close()
    
    return pil_image


def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Prompt Diffusion 이미지 생성기")
    parser.add_argument("--config", type=str, required=True, help="설정 파일 경로")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="모델 체크포인트 경로")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="베이스 모델 경로 또는 이름")
    parser.add_argument("--output_dir", type=str, default="./outputs/generated", help="출력 디렉토리")
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU 번호")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    parser.add_argument("--num_inference_steps", type=int, default=24, help="추론 스텝 수")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="분류기 없는 가이던스 스케일 (CFG)")
    parser.add_argument("--conditioning_scale", type=float, default=1.0, help="ControlNet 조건 스케일")
    parser.add_argument("--max_images", type=int, default=5000, help="생성할 최대 이미지 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--compute_fid", action="store_true", help="FID 계산용 이미지 저장 모드")
    parser.add_argument("--task", type=str, default=None, help="특정 태스크만 생성 (예: hed, depth, canny)")
    parser.add_argument("--black_support", action="store_true", default=None)
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
    
    train_tasks = data_config.get("train_tasks", ["canny", "depth", "hed", "normal"])
    if args.task:
        if args.task not in train_tasks:
            print(f"Warning: {args.task} is not in train_tasks. Adding it.")
            train_tasks.append(args.task)
        train_tasks = [args.task]
    
    print("Setting up data module...")
    datamodule = ControlDataModule(
        path=data_config.get("path", "/data2/david3684/ufg_diff/sd3control_base/datasets/laion_data/laion_nonhuman"),
        human_path=data_config.get("human_path", "/data2/david3684/ufg_diff/sd3control_base/datasets/laion_data/laion_human"),
        train_tasks=train_tasks,
        test_tasks=data_config.get("test_tasks", []),
        tasks_per_batch=1 if args.task else data_config.get("tasks_per_batch", 1),
        splits=(0.9, 0.1),
        res=512,
        shots=data_config.get("shots", 1),
        batch_size=args.batch_size,
        num_workers=data_config.get("num_workers", 2),
        total_samples=data_config.get("total_samples", 10000),
    )
    datamodule.setup()

    if args.compute_fid:
        test_ds = datamodule.val_ds
    else:
        indices = list(range(min(100, len(datamodule.val_ds))))
        test_ds = torch.utils.data.Subset(datamodule.val_ds, indices)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=data_config.get("num_workers", 2),
        shuffle=False,
    )
    
    model_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    ckpt_name = os.path.basename(args.checkpoint_path)
    if "." in ckpt_name:
        ckpt_name = ckpt_name.split(".")[0]
    
    output_dir = os.path.join(args.output_dir, model_name, ckpt_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for task in train_tasks:
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        if args.compute_fid:
            os.makedirs(os.path.join(task_dir, "fid"), exist_ok=True)
    
    # 이미지 생성 프로세스
    print("Starting image generation...")
    total_generated = 0
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating images")):
        if total_generated >= args.max_images:
            break
        
        images = batch["images"]              # [B, 2*shots, C, H, W]
        conditions = batch["conditions"]      # [B, T, 2*shots, C, H, W]
        prompts = batch["prompts"]            # List[List[str]] - [shots][B]
        task_indices = batch["task_indices"]  # [B, T]
        
        batch_size, num_tasks = task_indices.shape
        shots = images.shape[1] // 2  # 각 이미지에 대한 샷 수
        
        # 쿼리 및 서포트 컨디션/이미지 분리
        q_cond, sp_cond = torch.chunk(conditions, 2, dim=2)
        gt_image, sp_image = torch.chunk(images, 2, dim=1)
        
        for b in range(batch_size):
            for t in range(num_tasks):
                curr_task_idx = task_indices[b, t].item()
                curr_task_name = taskidx_to_taskname(curr_task_idx)
                
                for s in range(shots):
                    if total_generated >= args.max_images:
                        break
                    
                    # 현재 프롬프트
                    curr_prompt = prompts[s][b]
                    if not curr_prompt or curr_prompt.strip() == "":
                        curr_prompt = f"Generate an image based on the {curr_task_name} conditioning"
                    
                    # 현재 쿼리 및 서포트 이미지
                    query_control = q_cond[b, t, s].unsqueeze(0)         # [1, C, H, W]
                    if args.black_support:
                        support_control = torch.zeros_like(sp_cond[b, t, s].unsqueeze(0), device=device, dtype=torch.float16)
                        support_gt = torch.zeros_like(sp_image[b, s].unsqueeze(0), device=device, dtype=torch.float16)
                    else:
                        support_control = sp_cond[b, t, s].unsqueeze(0)       # [1, C, H, W]
                        support_gt = sp_image[b, s].unsqueeze(0)           # [1, C, H, W]
                    
                    with torch.no_grad():
                        generated = pipe(
                            prompt=curr_prompt,
                            height=512,
                            width=512,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            negative_prompt="lowres, low quality, worst quality, blurry",
                            control_image=query_control,
                            control_image_pair=[support_control, support_gt],
                            controlnet_conditioning_scale=args.conditioning_scale,
                            num_images_per_prompt=1,
                            generator=torch.Generator(device=device).manual_seed(args.seed + total_generated),
                        )
                    
                    # 생성된 이미지
                    generated_image = generated.images[0]
                    
                    if args.compute_fid:
                        fid_save_path = os.path.join(
                            output_dir, curr_task_name, "fid", f"{batch_idx:04d}_{b:02d}_{t:02d}_{s:02d}.png"
                        )
                        generated_image.save(fid_save_path)
                    else:
                        # 시각화 이미지 저장 디렉토리
                        vis_dir = os.path.join(output_dir, curr_task_name, "visualizations")
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # 원본 생성 이미지 저장 디렉토리
                        raw_dir = os.path.join(output_dir, curr_task_name, "raw_images")
                        os.makedirs(raw_dir, exist_ok=True)
                        
                        # 원본 이미지 저장 (black_support 여부 표시)
                        filename_suffix = "_black" if args.black_support else ""
                        raw_save_path = os.path.join(
                            raw_dir, f"{batch_idx:04d}_{b:02d}_{t:02d}_{s:02d}{filename_suffix}.png"
                        )
                        generated_image.save(raw_save_path)
                        
                        # 시각화를 위한 이미지 준비
                        query_control_np = query_control[0].permute(1, 2, 0).cpu().numpy()
                        
                        if args.black_support:
                            # 검정색 이미지로 시각화
                            support_control_np = np.zeros((512, 512, 3))
                            support_gt_np = np.zeros((512, 512, 3))
                        else:
                            support_control_np = sp_cond[b, t, s].permute(1, 2, 0).cpu().numpy()
                            support_gt_np = sp_image[b, s].permute(1, 2, 0).cpu().numpy()
                        
                        gt_img_np = gt_image[b, s].permute(1, 2, 0).cpu().numpy()
                        
                        # 시각화 저장 경로에 black_support 여부 표시
                        vis_filename_suffix = "black_" if args.black_support else ""
                        vis_save_path = os.path.join(
                            vis_dir, f"{batch_idx:04d}_{b:02d}_{t:02d}_{s:02d}_{vis_filename_suffix}vis.jpg"
                        )
                        
                        # 생성된 이미지 시각화
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
                        
                        # 디버깅 정보 출력
                        print(f"Generated image with {'black' if args.black_support else 'normal'} support")
                        print(f"Saved raw image to: {raw_save_path}")
                        print(f"Saved visualization to: {vis_save_path}")
                    
                    total_generated += 1
    
    print(f"Total Image Generated: {total_generated}")


if __name__ == "__main__":
    main()