import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
import torch
import numpy as np
import faiss
import csv
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from src.model import MedCLIPModel


def build_vector_database(csv_path, img_dir, weights_path, index_output, meta_output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting device for inference execution: {device}")

    model = MedCLIPModel(emb_dim=512)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{weights_path}' in the root directory.")
        return
    except RuntimeError as e:
        print(f"❌ Error: Checkpoint doesn't match model.py's architecture.\n{e}")
        return

    print("🔍 Pushing model to device...")
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"🔍 Looking for CSV data at {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"❌ FATAL: Cannot find {csv_path}. Check your data folder!")
        return

    try:
        with open(csv_path, mode='r', encoding='utf-8-sig', errors='ignore') as file:
            reader = csv.DictReader(file)
            data_rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception as e:
        print(f"❌ NATIVE FILE READ FAILED: {e}")
        return

    if len(data_rows) == 0:
        print("❌ Error: CSV file is empty or corrupted!")
        return

    # ROCO CSVs use 'PMC_ID' as the filename column. Fall back to it as the
    # ID too if a separate 'ROCO_ID' column doesn't exist.
    id_col = 'ROCO_ID' if 'ROCO_ID' in fieldnames else 'PMC_ID'
    print(f"🔍 CSV loaded with {len(data_rows)} rows ('{id_col}' used as ID). Starting extraction...")

    embeddings_list = []
    metadata_records = []

    print("\nStarting batch feature extraction loop...")
    with torch.no_grad():
        for row in tqdm(data_rows):
            img_id = row.get(id_col)
            img_name = row.get('PMC_ID')
            caption = row.get('Caption')

            if not img_id or not img_name or not caption:
                continue

            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_name = img_name + '.jpg'

            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            try:
                image = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(image).unsqueeze(0).to(device)

                img_embeddings = model.encode_image(img_tensor)

                embeddings_list.append(img_embeddings.cpu().numpy().flatten())
                metadata_records.append({
                    'image_id': img_id,
                    'image_path': img_path,
                    'caption': caption
                })
            except Exception:
                continue

    if len(embeddings_list) == 0:
        print("❌ Error: No images were successfully processed. Check your image paths.")
        return

    embeddings_matrix = np.array(embeddings_list).astype('float32')
    dimension = embeddings_matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)
    faiss.write_index(index, index_output)

    if metadata_records:
        keys = metadata_records[0].keys()
        with open(meta_output, 'w', newline='', encoding='utf-8-sig') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(metadata_records)

    print(f"\n🎉 Success! Database built completely.")
    print(f"Indexed instances: {len(metadata_records)}")
    print(f"Saved artifacts: '{index_output}' and '{meta_output}'")


if __name__ == "__main__":
    build_vector_database(
        csv_path="data/test_data.csv",
        img_dir="data/test_set",
        weights_path="medclip_expert_v2.pth",
        index_output="vector_index.faiss",
        meta_output="metadata.csv"
    )