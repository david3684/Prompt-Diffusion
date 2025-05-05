from typing import Any, Union
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import lightning as L
from glob import glob
import random
import os
import torchvision.transforms as T

TASKS = {
    "canny": 0,
    "depth": 1,
    "hed": 2,
    "normal": 3,
    "pose": 4, # representing segmentation task
    "densepose": 5,
}


class LaionBaseDataset(Dataset):
    def __init__(
        self,
        path: str,
        tasks: list[str],
        tasks_per_batch: int = 1,
        res: int = 512,
        shots: int = 1,
        indices: list = None,
        train: bool = True,
        shuffle: bool = True,
        task_map: dict = None,
    ):
        """
        Args:
            path: path to the directory containing the dataset: images dir, label1 dir, label2 dir, etc
            tasks: list of strings representing the tasks
            split: one of "train", "val", "test"
            splits: tuple of 3 floats representing the proportion of the dataset to use for train, val, and test
            res: resolution of the images
            shots: number of support pairs per query

        """
        self.path = path
        self.tasks = tasks
        self.tasks_per_batch = tasks_per_batch
        self.res = res
        self.shots = shots
        self.train = train
        self.shuffle = shuffle
        self.task_map = task_map 
        self.transform = T.Compose([
            T.Resize((res, res)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.condition_transform = T.Compose([
            T.Resize((res, res)),
            T.ToTensor()
        ])
        
        dirs = glob(self.path + "/*/")
        filenames = []
        for dir in dirs:
            filenames += glob(dir + "*.jpg")
        filenames.sort()
        # NOTE: keep it as it as on server 14 to make fair training experiments
        self.filenames = [filenames[i] for i in indices]
        print(f"Total number of files: {len(self.filenames)}")
        self.num_filegroups = len(self.filenames) // self.shots
        print(self.num_filegroups)

        if not shuffle:
            # deterministic grouping
            self.filegroups = [
                self.filenames[i : i + self.shots]
                for i in range(0, len(self.filenames), self.shots)
            ]
        else:
            # initialize filegroups for the shuffle case
            self.create_filegroups()

    def create_filegroups(self):
        # Shuffle filenames each time you want new filegroups
        shuffled = self.filenames.copy()
        random.shuffle(shuffled)

        # Group into filegroups
        self.filegroups = [
            shuffled[i : i + self.shots] for i in range(0, len(shuffled), self.shots)
        ]
        if len(self.filegroups[-1]) < self.shots:
            self.filegroups.pop(-1)       

    def __len__(self) -> int:
        return self.num_filegroups

    def __getitem__(self, i: int) -> dict[str, Any]:
        if i == 0 and self.shuffle:
            self.create_filegroups()

        sp_idx = np.random.randint(0, len(self.filegroups))
        while sp_idx == i and len(self) > 1:
            sp_idx = np.random.randint(0, len(self.filegroups))

        files = self.filegroups[i] + self.filegroups[sp_idx]

        # load images and normalize to 0~1
        images = [self.transform(Image.open(file).convert('RGB')) for file in files]
        # images = [torch.tensor(np.array(Image.open(file).convert('RGB'))).float() / 255. # Process by VAE
                    # for file in files]
        # images = [rearrange(image, 'h w c -> c h w') for image in images] # [2*shots, c, h, w]
        images = torch.stack(images) # [2*shots, c, h, w]
        
        # sample tasks
        if self.train:
            replace = self.tasks_per_batch > len(self.tasks)
            tasks = np.random.choice(self.tasks, self.tasks_per_batch, replace=replace)
        else:
            tasks = self.tasks
        task_indices = torch.tensor([self.task_map[t] for t in tasks]) # [T]

        # load conditions - keep these normalized to 0~1
        labels = []
        for task in tasks:
            task_labels = []
            for file in files:
                dir, name = file.split('/')[-2:]
                task_label = self.condition_transform(Image.open(f"{self.path}/{dir}/{task}/{name}").convert('RGB'))
                task_labels.append(task_label)
                # task_labels.append(torch.tensor(np.array(Image.open(f"{self.path}/{dir}/{task}/{name}").convert('RGB'))).float())
            # task_labels = [rearrange(label, 'h w c -> c h w') for label in task_labels]
            task_labels = torch.stack(task_labels) # [2*shots, c, h, w]
            labels.append(task_labels)
        labels = torch.stack(labels) # [T, 2*shots, c, h, w]

        # load text prompt
        prompts = []
        for file in files:
            prompt_path = file.replace('.jpg', '.txt')
            if os.path.exists(prompt_path):
                with open(prompt_path) as fp:
                    prompt = fp.read().strip()
                    prompts.append(prompt)
            else:
                prompts.append("")

        return dict(images=images, conditions=labels, prompts=prompts, task_indices=task_indices)


class CombineDatasets(Dataset):
    def __init__(self, datasets: list[Dataset], shuffle: bool = True):
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.total_size = sum(self.dataset_sizes)
        self.index_map = []

        # Build index mapping: (dataset_id, index_within_dataset)
        for i, dataset in enumerate(datasets):
            self.index_map += [(i, j) for j in range(len(dataset))]
        if shuffle:
            self.shuffle_indices()

    def shuffle_indices(self):
        random.shuffle(self.index_map)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        ds_id, idx = self.index_map[index]
        dataset = self.datasets[ds_id]
        item = dataset[idx]
        return item
    

class ControlDataModule(L.LightningDataModule):
    def __init__(
        self, 
        path: str,
        human_path: str,
        train_tasks: list[str],
        test_tasks: list[str],  
        tasks_per_batch: int = 1,
        splits: tuple[float, float] = (0.9, 0.1),
        res: int = 512,
        shots: int = 1,
        total_samples: int = 3e5,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.path = path
        self.human_path = human_path
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.tasks_per_batch = tasks_per_batch
        self.splits = splits
        self.res = res
        self.shots = shots
        self.total_samples = int(total_samples)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.current_train_batch_index = 0  # Track batch index
        
        all_tasks = list(dict.fromkeys(train_tasks + test_tasks))  # 중복 제거 및 순서 유지
        self.task_map = {task: idx for idx, task in enumerate(all_tasks)}
    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(1505)

        check_human_tasks = 'pose' in self.train_tasks and 'densepose' in self.train_tasks 

        # Init Human Dataset
        if check_human_tasks:
            total_samples = min(len(glob(self.human_path + '/*/*.jpg')), self.total_samples)
            train_indices, val_indices = torch.utils.data.random_split( # Need to split the indices to maintain the random task selection in getitem
                torch.arange(total_samples), self.splits, generator=generator
            )
            print(len(train_indices))
            train_human_ds = LaionBaseDataset(
                path=self.human_path,
                tasks=['pose', 'densepose'],
                tasks_per_batch=self.tasks_per_batch,
                shots=self.shots,
                res=self.res,
                indices=train_indices,
                train=True,
                task_map = self.task_map,
            )
            # val_human_ds = LaionBaseDataset(
            #     path=self.human_path,
            #     tasks=['pose', 'densepose'],
            #     tasks_per_batch=self.tasks_per_batch,
            #     shots=self.shots,
            #     res=self.res,
            #     indices=val_indices,
            #     train=False
            # )

        # Init 2nd Dataset
        total_samples = min(len(glob(self.path + '/*/*.jpg')), self.total_samples)
        print(total_samples)
        train_indices, val_indices = torch.utils.data.random_split( # Need to split the indices to maintain the random task selection in getitem
            torch.arange(total_samples), self.splits, generator=generator
        )
        print(len(train_indices))
        train_tasks = self.train_tasks.copy()
        if check_human_tasks:
            train_tasks.remove('pose')
            train_tasks.remove('densepose')
        train_normal_ds = LaionBaseDataset(
            path=self.path,
            tasks=train_tasks,
            tasks_per_batch=self.tasks_per_batch,
            shots=self.shots,
            res=self.res,
            indices=train_indices,
            train=True,
            task_map = self.task_map,
        )
        # val_normal_ds = LaionBaseDataset(
        #     path=self.path,
        #     tasks=train_tasks,
        #     tasks_per_batch=self.tasks_per_batch,
        #     shots=self.shots,
        #     res=self.res,
        #     indices=val_indices,
        #     train=False
        # )

        # Init Combined Dataset
        if check_human_tasks:
            self.train_ds = CombineDatasets([train_normal_ds, train_human_ds], shuffle=True)
            # self.val_ds = CombineDatasets([val_normal_ds, val_human_ds])
        else:
            # If no human dataset, just use the normal dataset
            self.train_ds = train_normal_ds
            # self.val_ds = val_normal_ds

        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
        )


    def tuning_dataloader(
            self,
            tasks: Union[str, list[str]],
            num_supports: int = 15,
            shots: int = 1,):
        indices = range(int(self.total_samples//2) + 15+ 24, int(self.total_samples//2) + 15 + 24 + num_supports)
        if isinstance(tasks, str):
            tasks = [tasks]
        human_task = 'pose' in tasks or 'densepose' in tasks
        self.tuning_ds = LaionBaseDataset(
            path=self.human_path if human_task else self.path,
            tasks=tasks,
            tasks_per_batch=1,
            shots=shots,
            res=self.res,
            indices=indices,
            train=False,
            task_map = self.task_map,
            shuffle=True, # Fix support set for inference visualization, the data is shuffle in training code
        )
        return DataLoader(
            self.tuning_ds,
            batch_size=1,
            num_workers=4,
        )

