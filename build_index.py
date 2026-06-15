import os
import torch
import numpy as np
import faiss
import csv
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Import your custom architecture blueprint
from src.model import MedCLIPModel

def build_vector_database(csv_path, img_dir, weights_path, index_output, meta_output):
    # Set execution target to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting device for inference execution: {device}")
    
    # 1. Initialize Model Architecture and Bind Trained Weights
    model = MedCLIPModel(emb_dim=512)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{weights_path}' in the root directory.")
        return
        
    print("🔍 Pushing model to GPU VRAM...")
    model.to(device)
    model.eval()
    
    # 2. Configure Evaluation Image Preprocessing Pipelines
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Target Data Indexes using PURE PYTHON
    print(f"🔍 Looking for CSV data at {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"❌ FATAL: Cannot find {csv_path}. Check your data folder!")
        return
        
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig', errors='ignore') as file:
            reader = csv.DictReader(file)
            data_rows = list(reader)
    except Exception as e:
        print(f"❌ NATIVE FILE READ FAILED: {e}")
        return
        
    if len(data_rows) == 0:
        print("❌ Error: CSV file is empty or corrupted!")
        return
        
    print(f"🔍 CSV loaded successfully with {len(data_rows)} rows. Starting extraction...")
    
    embeddings_list = []
    metadata_records = []
    
    print("\nStarting batch feature extraction loop...")
    with torch.no_grad():
        for row in tqdm(data_rows):
            
            # Extract exactly what we need
            img_id = row.get('ROCO_ID')
            img_name = row.get('PMC_ID') # This contains the actual filename
            caption = row.get('Caption')
            
            if not img_id or not img_name or not caption:
                continue # Skip empty lines
            
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue  # Skip missing file instances gracefully
                
            try:
                # Load image and map to tensor shape
                image = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Compute visual vector embedding
                img_features = model.vision_model(img_tensor)
                img_embeddings = model.vision_projection(img_features)
                
                # Normalize onto unit sphere for accurate cosine matching
                img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
                
                # Append arrays for index generation
                embeddings_list.append(img_embeddings.cpu().numpy().flatten())
                metadata_records.append({
                    'image_id': img_id,
                    'image_path': img_path,
                    'caption': caption
                })
            except Exception as e:
                # Skip corrupt images without breaking execution run
                continue
                
    if len(embeddings_list) == 0:
        print("❌ Error: No images were successfully processed. Check your image paths.")
        return

    # 4. Compile Array Matrices into structural FAISS Space
    embeddings_matrix = np.array(embeddings_list).astype('float32')
    dimension = embeddings_matrix.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)
    faiss.write_index(index, index_output)
    
    # 5. Save Metadata using PURE PYTHON (No Pandas)
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
        weights_path="medclip_expert_v2.pth", # <--- Updated to Expert Weights
        index_output="vector_index.faiss",
        meta_output="metadata.csv"
    )