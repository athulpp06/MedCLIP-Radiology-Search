import os
import torch
import pandas as pd
import numpy as np
import faiss
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
        
    model.to(device)
    model.eval()
    
    # 2. Configure Evaluation Image Preprocessing Pipelines
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Target Data Indexes
    if not os.path.exists(csv_path):
        print(f"❌ Error: Metadata CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    embeddings_list = []
    metadata_records = []
    
    print("\nStarting batch feature extraction loop...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_id = row['ImageIndex']
            caption = row['Caption']
            
            img_name = f"{img_id}.jpg"
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue  # Skip missing file instances gracefully
                
            try:
                # Load image and map to tensor shape
                image = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Compute visual vector embedding
                img_features = model.vision_encoder(img_tensor)
                img_embeddings = model.image_projection(img_features)
                
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
        print("❌ Error: No images were successfully processed. Check your paths.")
        return

    # 4. Compile Array Matrices into structural FAISS Space
    embeddings_matrix = np.array(embeddings_list).astype('float32')
    dimension = embeddings_matrix.shape[1]
    
    # Flat Inner Product index architecture works perfectly for normalized vectors
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)
    
    # 5. Save Artifact Files to Local Storage
    faiss.write_index(index, index_output)
    meta_df = pd.DataFrame(metadata_records)
    meta_df.to_csv(meta_output, index=False)
    
    print(f"\n🎉 Success! Database built completely.")
    print(f"Indexed instances: {len(metadata_records)}")
    print(f"Saved artifacts: '{index_output}' and '{meta_output}'")

if __name__ == "__main__":
    # Point these to your local data paths
    build_vector_database(
        csv_path="data/test_captions.csv", 
        img_dir="data/test_set",
        weights_path="medclip_v1_weights.pth",
        index_output="vector_index.faiss",
        meta_output="metadata.csv"
    )