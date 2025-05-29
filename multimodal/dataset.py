import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
# get from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imagefx import crop, preprocess  # ← 여기 추가


COLUMNS_TO_DROP = ['R_VA', 'L_PICA', 'R_PICA', 'L_SCA', 'R_SCA', 'L_PCA', 'R_PCA'] # less than 1% of the data

class AneurysmDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, transform=None, train_type='train'):
        
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        self.train_type = train_type  # 'train' or 'test'
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

        images = []
        for suffix in self.image_order:
            #print(suffix)
            image_path = os.path.join(self.image_dir, f"{patient_id}{suffix}.jpg")
            image = Image.open(image_path).convert("RGB")

            # 이미지 전처리: crop → invert/sharpen/contrast → transform
        
            image = crop(image)
         
            image = preprocess(image)
            
            if self.transform:
                image = self.transform(image)

            images.append(image)

        # 이미지 텐서 스택: [8, 3, H, W]
        images = torch.stack(images)

        # 텍스트 리스트: length 8, 각 이미지에 대한 설명
        texts = [self.text_map[suffix] for suffix in self.image_order]

        # 라벨: [22] float tensor
        # drop columns
        row = row.drop(COLUMNS_TO_DROP)
   
        label = torch.tensor(row.values[1:], dtype=torch.float)

        return images, texts, label
