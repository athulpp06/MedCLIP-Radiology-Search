import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel

class MedCLIPModel(nn.Module):
    def __init__(self, emb_dim=512):
        super(MedCLIPModel, self).__init__()
        
        # 1. Vision Model (Matches Kaggle 'vision_model')
        self.vision_model = models.resnet50(weights=None)
        num_ftrs = self.vision_model.fc.in_features
        self.vision_model.fc = nn.Identity() 
        
        # Matches Kaggle 'vision_projection'
        self.vision_projection = nn.Linear(num_ftrs, emb_dim)

        # 2. Text Model (Matches Kaggle 'text_model')
        self.text_model_name = "emilyalsentzer/Bio_ClinicalBERT" # Make sure this matches your Kaggle model!
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        
        # Matches Kaggle 'text_projection'
        text_hidden_size = self.text_model.config.hidden_size
        self.text_projection = nn.Linear(text_hidden_size, emb_dim)
        
        # Matches Kaggle 'logit_scale' (Required to load weights successfully)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, image, input_ids, attention_mask):
        image_features = self.vision_model(image)
        image_embeddings = self.vision_projection(image_features)
        
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs.last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_features)
        
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return image_embeddings, text_embeddings