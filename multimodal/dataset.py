import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# csv_path = "/home/edlab/sjim/k-ium-coding-vessels/train_set/train.csv"
# image_dir = "/home/edlab/sjim/k-ium-coding-vessels/train_set/images"
class AneurysmDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        self.image_order = ['LI-A', 'LI-B', 'RI-A', 'RI-B', 'LV-A', 'LV-B', 'RV-A', 'RV-B']
        self.text_map = {
            'LI-A': "Left internal carotid artery injection, view A",
            'LI-B': "Left internal carotid artery injection, view B",
            'RI-A': "Right internal carotid artery injection, view A",
            'RI-B': "Right internal carotid artery injection, view B",
            'LV-A': "Left vertebral artery injection, view A",
            'LV-B': "Left vertebral artery injection, view B",
            'RV-A': "Right vertebral artery injection, view A",
            'RV-B': "Right vertebral artery injection, view B",
        }
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["Index"])
        
        # 1. 이미지 로딩
        # imagefx에서 crop, preprocess사용하기
        images = []
        for suffix in self.image_order:
            image_path = os.path.join(self.image_dir, f"{patient_id}{suffix}.jpg")
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)  # [8, 3, H, W]

        # 2. 텍스트 설명 리스트
        texts = [self.text_map[suffix] for suffix in self.image_order]

        # 3. 레이블
        label = torch.tensor(row.values[1:], dtype=torch.float)  # [22]
        
        return images, texts, label
