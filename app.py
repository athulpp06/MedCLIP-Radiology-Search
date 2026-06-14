import streamlit as st
import torch
import pandas as pd
import faiss
import numpy as np
from PIL import Image
import os

# Import your custom architecture blueprint
from src.model import MedCLIPModel

# 1. Configure the Web Interface
st.set_page_config(page_title="MedCLIP Search", layout="wide")
st.title("🔍 MedCLIP Semantic Search Engine")
st.markdown("Search through thousands of radiology images using natural language clinical descriptions.")

# 2. Load the AI and Database (Cached so it only happens once)
@st.cache_resource
def load_search_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Architecture and Weights
    model = MedCLIPModel(emb_dim=512)
    model.load_state_dict(torch.load("medclip_v1_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Load the FAISS Vector Index and Metadata
    index = faiss.read_index("vector_index.faiss")
    df = pd.read_csv("metadata.csv")
    
    return model, index, df, device

with st.spinner("Loading AI model and vector database into GPU memory..."):
    try:
        model, index, df, device = load_search_engine()
    except Exception as e:
        st.error(f"❌ Failed to load systems. Check your file paths! Error: {e}")
        st.stop()

# 3. The Search Bar
query = st.text_input("Enter a clinical feature to search for (e.g., 'pleural effusion', 'axial MRI', 'cardiomegaly'):", "")

# 4. The Inference Pipeline
if query:
    with st.spinner("Embedding query and scanning database..."):
        with torch.no_grad():
            # Convert text into machine tokens
            tokens = model.tokenizer(
                [query], 
                padding=True, 
                truncation=True, 
                max_length=77, 
                return_tensors="pt"
            ).to(device)
            
            # Push tokens through the Text Model (This matches our src/model.py fix!)
            text_outputs = model.text_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
            
            # Extract the raw features
            if hasattr(text_outputs, 'pooler_output'):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs.last_hidden_state[:, 0, :]
            
            # Project to the shared 512-dimensional space and normalize
            text_embeddings = model.text_projection(text_features)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_vector = text_embeddings.cpu().numpy().astype('float32')
        
        # 5. Search the FAISS Index
        k = 6 # Number of images to return
        distances, indices = index.search(text_vector, k)
        
        # 6. Render the Results
        st.divider()
        st.subheader(f"Top {k} highest-matching scans for: *'{query}'*")
        
        # Create a clean 3-column grid layout
        cols = st.columns(3)
        for i in range(k):
            idx = indices[0][i]
            score = distances[0][i]
            row = df.iloc[idx]
            
            img_path = row['image_path']
            original_caption = row['caption']
            
            with cols[i % 3]:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    # Display the FAISS cosine similarity score
                    st.caption(f"**Cosine Similarity:** {score:.4f}")
                    # Hide the original caption inside a dropdown to keep the UI clean
                    with st.expander("View Original Dataset Caption"):
                        st.write(original_caption)
                except FileNotFoundError:
                    st.error(f"Image missing at: {img_path}")