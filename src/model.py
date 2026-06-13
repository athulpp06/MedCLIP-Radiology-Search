import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel

class MedCLIPModel(nn.Module):
    def __init__(self, emb_dim=512):
        super(MedCLIPModel, self).__init__()
        
        # 1. Vision Encoder: ResNet50
        # Weights=None because we will load your custom-trained .pth checkpoint weights
        self.vision_encoder = models.resnet50(weights=None)
        num_ftrs = self.vision_encoder.fc.in_features
        self.vision_encoder.fc = nn.Identity() # Strip away final classification layer
        
        # Image Projection Layer (Maps ResNet features to shared 512-dim space)
        self.image_projection = nn.Sequential(
            nn.Linear(num_ftrs, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # 2. Text Encoder: ClinicalBERT / BioLinkBERT
        self.text_model_name = "AnatolyB/BioGpt-BioLinkBERT-base" # Change if using a different HuggingFace path
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_encoder = AutoModel.from_pretrained(self.text_model_name)
        
        # Text Projection Layer (Maps BERT hidden states to shared 512-dim space)
        text_hidden_size = self.text_encoder.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, image, input_ids, attention_mask):
        # Extract visual vectors
        image_features = self.vision_encoder(image)
        image_embeddings = self.image_projection(image_features)
        
        # Extract textual vectors
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler_output or CLS token state
        text_features = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs.last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_features)
        
        # Normalize representations to unit spheres for accurate cosine similarity
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return image_embeddings, text_embeddings