import streamlit as st
import torch
import pandas as pd
import faiss
import numpy as np
from PIL import Image
import os

from src.model import MedCLIPModel

st.set_page_config(page_title="MedCLIP Search", layout="wide")
st.title("🔍 MedCLIP Semantic Search Engine")
st.markdown("Search through radiology images using natural language clinical descriptions.")


@st.cache_resource
def load_search_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MedCLIPModel(emb_dim=512)
    model.load_state_dict(torch.load("medclip_expert_v2.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    index = faiss.read_index("vector_index.faiss")
    df = pd.read_csv("metadata.csv")

    return model, index, df, device


with st.spinner("Loading AI model and vector database..."):
    try:
        model, index, df, device = load_search_engine()
    except Exception as e:
        st.error(f"❌ Failed to load systems. Check your file paths! Error: {e}")
        st.stop()

query = st.text_input(
    "Enter a clinical feature to search for (e.g., 'pleural effusion', 'axial MRI', 'cardiomegaly'):",
    ""
)

if query:
    with st.spinner("Embedding query and scanning database..."):
        with torch.no_grad():
            tokens = model.tokenizer(
                [query],
                padding='max_length',
                truncation=True,
                max_length=128,  # matches training max_length exactly
                return_tensors="pt"
            ).to(device)

            # Mean-pools over real tokens — must match training, or the
            # projection head's output won't mean what it was trained to mean.
            text_embeddings = model.encode_text(tokens['input_ids'], tokens['attention_mask'])
            text_vector = text_embeddings.cpu().numpy().astype('float32')

        k = 6
        distances, indices = index.search(text_vector, k)

        st.divider()
        st.subheader(f"Top {k} highest-matching scans for: *'{query}'*")

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
                    st.caption(f"**Cosine Similarity:** {score:.4f}")
                    with st.expander("View Original Dataset Caption"):
                        st.write(original_caption)
                except FileNotFoundError:
                    st.error(f"Image missing at: {img_path}")