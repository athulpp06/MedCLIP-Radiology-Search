import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from transformers import AutoTokenizer, AutoModel


class MedCLIPModel(nn.Module):
    """
    Dual-encoder MedCLIP model. This MUST match the Kaggle training notebook's
    architecture exactly (layer names, shapes, pooling strategy), or
    load_state_dict() will fail or silently drop weights.

    Matches the FINAL hybrid training run: MLP projection heads, mean-pooled
    text features, clamped learnable temperature. NOT the original
    single-Linear / CLS-token version.
    """
    def __init__(self, emb_dim=512):
        super(MedCLIPModel, self).__init__()

        # weights=None: we're loading our own trained weights next,
        # no need to download ImageNet weights first.
        self.vision_model = models.resnet50(weights=None)
        num_ftrs = self.vision_model.fc.in_features
        self.vision_model.fc = nn.Identity()

        self.vision_projection = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, emb_dim)
        )

        self.text_model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)

        text_hidden_size = self.text_model.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, emb_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Inference-only deployment — keep gradients off even if .train() is
        # accidentally called somewhere.
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.text_model.parameters():
            p.requires_grad = False

    def get_logit_scale(self):
        return torch.clamp(self.logit_scale, max=np.log(100)).exp()

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean-pool over real (non-padding) tokens. This MUST match training —
        using a different pooling strategy here than what the projection head
        was trained on produces wrong embeddings with no error message at all.
        """
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return summed / counts

    def encode_image(self, image):
        features = self.vision_model(image)
        embeddings = self.vision_projection(features)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        features = self.mean_pooling(outputs, attention_mask)
        embeddings = self.text_projection(features)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def forward(self, image, input_ids, attention_mask):
        image_embeddings = self.encode_image(image)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        return image_embeddings, text_embeddings