import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class ROCODataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer, transform=None, max_length=128):
        """
        Expects the ROCO-style CSV columns 'PMC_ID' (image filename, no
        extension needed) and 'Caption'. These are the actual column names
        used by the Kaggle training CSVs.
        """
        COL_ID = 'PMC_ID'
        COL_CAPTION = 'Caption'

        self.df = pd.read_csv(csv_path)

        missing = [c for c in (COL_ID, COL_CAPTION) if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Expected column(s) {missing} not found in {csv_path}. "
                f"Available columns: {list(self.df.columns)}"
            )

        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.col_id = COL_ID
        self.col_caption = COL_CAPTION

        valid_indices = []
        for idx, row in self.df.iterrows():
            img_name = self._resolve_filename(str(row[self.col_id]))
            if os.path.exists(os.path.join(self.img_dir, img_name)):
                valid_indices.append(idx)

        dropped = len(self.df) - len(valid_indices)
        if dropped > 0:
            print(f"⚠️  {dropped} of {len(self.df)} rows skipped (image not found in {img_dir})")

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    @staticmethod
    def _resolve_filename(img_id):
        if not img_id.lower().endswith(('.jpg', '.jpeg', '.png')):
            return img_id + '.jpg'
        return img_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = self._resolve_filename(str(row[self.col_id]))
        caption = str(row[self.col_caption])

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'caption': caption
        }