from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

TASK_MAPPING = {
    'pose': 'human',
    'densepose': 'human',
    'canny': 'nonhuman',
    'depth': 'nonhuman',
    'hed': 'nonhuman',
    'normal': 'nonhuman',
}

class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        task_list = ['canny', 'depth', 'hed', 'normal'],
        split: str = "train",
        splits: tuple[float, float] = (0.9, 0.1), 
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        prompt_option: str = 'edit',
        max_samples_per_task: int = 150000, 
    ):
        assert split in ("train", "val")  
        # assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.task_list = task_list
        self.max_samples_per_task = max_samples_per_task
        self.prompt_option = prompt_option
        self.split = split
        # import ipdb; ipdb.set_trace()
        train_ratio, val_ratio = splits
        
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.condition_transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        
        self.file_mapping = {task: [] for task in task_list}
        
        for task in task_list:
            folder_type = TASK_MAPPING[task]
            base_path = os.path.join(self.path, f"laion_{folder_type}")
            
            files = []
            subdirs = glob(os.path.join(base_path, "*/"))
            for subdir in subdirs:
                img_files = glob(os.path.join(subdir, "*.jpg"))
                for img_file in img_files:
                    filename = os.path.basename(img_file)
                    dir_name = os.path.basename(os.path.dirname(img_file))
                    
                    control_path = os.path.join(base_path, dir_name, task, filename)
                    txt_path = img_file.replace('.jpg', '.txt')
                    
                    files.append({
                        'gt_path': img_file,
                        'dir_name': dir_name,
                        'filename': filename,
                        'control_path': control_path,
                        'txt_path': txt_path,
                        'base_path': base_path
                    })
            
            files.sort(key=lambda x: x['gt_path'])
            
            if len(files) > self.max_samples_per_task:
                files = files[:self.max_samples_per_task]
            
            if split == "train":
                idx_end = math.floor(train_ratio * len(files))
                self.file_mapping[task] = files[:idx_end]
            else:  # val
                idx_start = math.floor(train_ratio * len(files))
                self.file_mapping[task] = files[idx_start:]
            
            print(f"Task {task}: {len(self.file_mapping[task])} {split} samples")
        
        self.max_task_size = max([len(files) for files in self.file_mapping.values()], default=0)
        print(f"EditDataset loaded: using max task size of {self.max_task_size} samples")
            
    def __len__(self) -> int:
        return self.max_task_size

    def __getitem__(self, i: int) -> dict[str, Any]:
        task = np.random.choice(self.task_list)
        
        task_files = self.file_mapping[task]
        
        file_idx = i % len(task_files)
        file_data = task_files[file_idx]
        
        gt_path = file_data['gt_path']
        base_path = file_data['base_path'] 
        dir_name = file_data['dir_name']
        filename = file_data['filename']
        control_query_path = file_data['control_path']
        txt_path = file_data['txt_path']
        
        same_folder_files = [f for f in self.file_mapping[task] 
                            if f['dir_name'] == dir_name and f['gt_path'] != gt_path]
        
        support_data = np.random.choice(same_folder_files)
        
        support_file = support_data['gt_path']
        support_filename = support_data['filename']
        control_support_path = os.path.join(base_path, dir_name, task, support_filename)
        
        image_q = Image.open(gt_path).convert('RGB')   
        image_q = self.transform(image_q)     
        # image_q = 2 * torch.tensor(np.array(image_q)).float() / 255. - 1  
        
        image_sp = Image.open(support_file).convert('RGB')
        image_sp = self.transform(image_q)     
        # image_sp = 2 * torch.tensor(np.array(image_sp)).float() / 255. - 1  # [-1, 1] 
        
        with open(txt_path, 'r') as f:
            prompt = f.read().strip()
        
        txt_log = task
        
        control_q = Image.open(control_query_path).convert('RGB')
        control_q = self.condition_transform(control_q)
        # control_q = torch.tensor(np.array(control_q)).float() / 255.0  # [0, 1] 
            
        control_sp = Image.open(control_support_path).convert('RGB')
        control_sp = self.condition_transform(control_sp)
        # control_sp = torch.tensor(np.array(control_sp)).float() / 255.0  # [0, 1] 
        
        # example_pair 
        example_pair = torch.cat((control_sp, image_sp), dim=2)  # h w c
        image_query = control_q
        image_target = image_q
        
        return dict(jpg=image_target, txt=prompt, query=image_query, example_pair=example_pair, txt_log=txt_log)

