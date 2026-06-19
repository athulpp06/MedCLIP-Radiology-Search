"""
Computes Recall@1/5/10 and MRR on the local TEST set — data the model never
saw during training or checkpoint selection (unlike the val set, which was
used to pick the best epoch). This is the cleanest, most defensible number
for a report.

Usage: python evaluate_test_set.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model import MedCLIPModel
from src.dataset import ROCODataset


def main(
    csv_path="data/test_data.csv",
    img_dir="data/test_set",
    weights_path="medclip_expert_v2.pth",
    max_samples=1000,
    batch_size=32,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MedCLIPModel(emb_dim=512)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ROCODataset(csv_path, img_dir, model.tokenizer, transform=transform)
    print(f"Test set size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_image_embeds, all_text_embeds = [], []
    num_samples = 0

    print(f"Extracting embeddings (cap: {max_samples})...")
    with torch.no_grad():
        for batch in loader:
            if num_samples >= max_samples:
                break
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            all_image_embeds.append(model.encode_image(images).cpu())
            all_text_embeds.append(model.encode_text(input_ids, attention_mask).cpu())
            num_samples += images.size(0)

    image_embeddings = torch.cat(all_image_embeds, dim=0)[:max_samples]
    text_embeddings = torch.cat(all_text_embeds, dim=0)[:max_samples]

    similarity_matrix = (text_embeddings @ image_embeddings.T).numpy()
    num_queries = similarity_matrix.shape[0]

    mrr_sum = 0.0
    correct_top_1 = correct_top_5 = correct_top_10 = 0
    ranks = []

    for i in range(num_queries):
        ranked_indices = np.argsort(-similarity_matrix[i])
        rank = int(np.where(ranked_indices == i)[0][0]) + 1
        ranks.append(rank)
        mrr_sum += 1.0 / rank
        if rank == 1: correct_top_1 += 1
        if rank <= 5: correct_top_5 += 1
        if rank <= 10: correct_top_10 += 1

    print("-" * 40)
    print(f"TEST SET — {num_queries} pairs (never used for training/checkpoint selection)")
    print(f"Recall@1:  {(correct_top_1 / num_queries) * 100:05.2f}%")
    print(f"Recall@5:  {(correct_top_5 / num_queries) * 100:05.2f}%")
    print(f"Recall@10: {(correct_top_10 / num_queries) * 100:05.2f}%")
    print(f"MRR:       {mrr_sum / num_queries:.4f}")
    print(f"Median rank: {np.median(ranks):.0f} / {num_queries}")
    print("-" * 40)

    np.save("test_image_embeddings.npy", image_embeddings.numpy())
    np.save("test_text_embeddings.npy", text_embeddings.numpy())
    print("Saved test_image_embeddings.npy / test_text_embeddings.npy")


if __name__ == "__main__":
    main()