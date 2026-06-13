import streamlit as st
import torch
import pandas as pd
import faiss
import os
from PIL import Image

# Import your custom architecture blueprint
from src.model import MedCLIPModel

# Page configuration layout properties
st.set_page_config(page_title="Med-CLIP Explorer", page_icon="🩻", layout="wide")

@st.cache_resource
def load_search_engine():
    """Loads and caches runtime structures to prevent reloads on user actions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and inject weights
    model = MedCLIPModel(emb_dim=512)
    model.load_state_dict(torch.load("medclip_v1_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Load database matrices
    index = faiss.read_index("vector_index.faiss")
    metadata = pd.read_csv("metadata.csv")
    
    return model, index, metadata, device

# Header UI Elements
st.title("🩻 Med-CLIP: Medical Image Semantic Search Engine")
st.markdown("Query an offline database of chest radiographs using free-text natural language descriptions.")
st.markdown("---")

# Setup error checkpointing interface
try:
    model, index, metadata, device = load_search_engine()
    st.sidebar.success(f"⚙️ Running Engine on: {str(device).upper()}")
    st.sidebar.info(f"📚 Total Images Searchable: {len(metadata)}")
except Exception as e:
    st.sidebar.error("❌ Dependencies Missing")
    st.warning("Before launching: You must generate your index files by running: `python build_index.py`")
    st.stop()

# Layout Design: Input controls
query_text = st.text_input(
    "Enter clinical observations or suspected pathologies:", 
    placeholder="e.g., pleural effusion in left lung field or normal chest radiograph without abnormalities"
)

top_k = st.sidebar.slider("Maximum matches to return:", min_value=1, max_value=9, value=3, step=1)

if query_text:
    with st.spinner("Traversing latent vector alignment space..."):
        # 1. Encode text string to structural embedding
        tokens = model.tokenizer(
            [query_text], 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            text_features = model.text_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
            # Match pooler mechanism strategy defined inside model.py
            text_features = text_outputs = text_features.pooler_output if hasattr(text_features, 'pooler_output') else text_features.last_hidden_state[:, 0, :]
            text_embeddings = model.text_projection(text_features)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_query_vector = text_embeddings.cpu().numpy().astype('float32')
            
        # 2. Run instant vector search computation
        scores, indices = index.search(text_query_vector, top_k)
        
        # 3. Render Visual Grid Results
        st.subheader(f"Top {top_k} Most Similar Clinical Instances")
        columns_grid = st.columns(3)
        
        for grid_idx, (score, match_row_idx) in enumerate(zip(scores[0], indices[0])):
            if match_row_idx < 0 or match_row_idx >= len(metadata):
                continue
                
            record = metadata.iloc[match_row_idx]
            local_img_path = record['image_path']
            ground_truth_text = record['caption']
            
            # Balance display across 3 rows iteratively
            column_target = columns_grid[grid_idx % 3]
            
            with column_target:
                st.markdown(f"**Match Relevance Rank #{grid_idx + 1}**")
                st.caption(f"Cosine Proximity Score: `{score:.4f}`")
                
                if os.path.exists(local_img_path):
                    loaded_img = Image.open(local_img_path)
                    st.image(loaded_img, use_container_width=True)
                else:
                    st.error(f"Image asset file missing at path: {local_img_path}")
                
                with st.expander("View Original Clinical Caption"):
                    st.write(ground_truth_text)