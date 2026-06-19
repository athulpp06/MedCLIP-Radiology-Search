import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
print("torch imported", flush=True)

from src.model import MedCLIPModel
print("MedCLIPModel imported", flush=True)

device = torch.device('cuda')
model = MedCLIPModel(emb_dim=512)
print("model object created", flush=True)

model = model.to(device)
print("model moved to cuda", flush=True)

print("loading checkpoint...", flush=True)
state_dict = torch.load("medclip_expert_v2.pth", map_location=device, weights_only=True)
print("checkpoint loaded into memory", flush=True)

model.load_state_dict(state_dict)
print("state dict applied - SUCCESS", flush=True)