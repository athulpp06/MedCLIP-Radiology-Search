import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ROCODataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer, transform=None, max_length=128):
        """
        Custom Dataset for loading Radiology Images and Clinical Captions.
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        
        # Prefilter to ensure we only keep rows where images actually exist locally
        valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, f"{row['ImageIndex']}.jpg")
            if os.path.exists(img_path):
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['ImageIndex']
        caption = str(row['Caption'])

        # 1. Load and Transform Image
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # 2. Tokenize text findings
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Squeeze out the extra batch dimension added by the tokenizer default return
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'caption': caption
        }