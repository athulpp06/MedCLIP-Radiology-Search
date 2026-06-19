"""
Regenerates analysis charts and Recall@K/MRR locally from your Kaggle output
files — no GPU or retraining needed.

Put these 4 files in ./kaggle_outputs/ first:
    medclip_training_history.csv
    medclip_error_log.csv
    val_image_embeddings.npy
    val_text_embeddings.npy

Produces, in ./analysis_output/:
    loss_curve.png, accuracy_curve.png, error_rank_distribution.png, summary.txt
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = "kaggle_outputs"
OUT_DIR = "analysis_output"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    history = pd.read_csv(os.path.join(IN_DIR, "medclip_training_history.csv"))
    error_log = pd.read_csv(os.path.join(IN_DIR, "medclip_error_log.csv"))
    image_embeddings = np.load(os.path.join(IN_DIR, "val_image_embeddings.npy"))
    text_embeddings = np.load(os.path.join(IN_DIR, "val_text_embeddings.npy"))

    plt.figure(figsize=(8, 4.5))
    plt.plot(history['epoch'], history['train_loss'], label='Train loss', marker='o', markersize=3)
    plt.plot(history['epoch'], history['val_loss'], label='Val loss', marker='o', markersize=3, linestyle='--')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(history['epoch'], history['train_acc'] * 100, label='Train acc', marker='o', markersize=3)
    plt.plot(history['epoch'], history['val_acc'] * 100, label='Val acc', marker='o', markersize=3, linestyle='--')
    plt.xlabel("Epoch"); plt.ylabel("In-batch accuracy (%)"); plt.title("Training & Validation Accuracy")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_curve.png"), dpi=150); plt.close()

    similarity_matrix = text_embeddings @ image_embeddings.T
    num_queries = similarity_matrix.shape[0]
    mrr_sum, r1, r5, r10 = 0.0, 0, 0, 0
    ranks = []
    for i in range(num_queries):
        ranked = np.argsort(-similarity_matrix[i])
        rank = int(np.where(ranked == i)[0][0]) + 1
        ranks.append(rank)
        mrr_sum += 1.0 / rank
        if rank == 1: r1 += 1
        if rank <= 5: r5 += 1
        if rank <= 10: r10 += 1

    recall_1, recall_5, recall_10 = r1/num_queries*100, r5/num_queries*100, r10/num_queries*100
    mrr = mrr_sum / num_queries

    plt.figure(figsize=(8, 4.5))
    bins = [1, 2, 5, 10, 20, 50, 100, num_queries + 1]
    labels = ['rank 2', 'rank 3-5', 'rank 6-10', 'rank 11-20', 'rank 21-50', 'rank 51-100', 'rank 100+']
    error_log['bucket'] = pd.cut(error_log['actual_rank'], bins=bins, labels=labels, right=True, include_lowest=True)
    counts = error_log['bucket'].value_counts().reindex(labels)
    plt.bar(labels, counts.values, color='#D85A30')
    plt.ylabel("Number of queries"); plt.title(f"Where the {len(error_log)} non-top-1 queries ranked")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_rank_distribution.png"), dpi=150); plt.close()

    summary = f"""MedCLIP Evaluation Summary
{'=' * 40}
Total epochs trained: {len(history)}
Final train accuracy: {history['train_acc'].iloc[-1] * 100:.2f}%
Final val accuracy:   {history['val_acc'].iloc[-1] * 100:.2f}%

Retrieval benchmark ({num_queries} validation pairs):
Recall@1:  {recall_1:.2f}%
Recall@5:  {recall_5:.2f}%
Recall@10: {recall_10:.2f}%
MRR:       {mrr:.4f}
Median rank: {np.median(ranks):.0f} / {num_queries}

Chance baseline ({num_queries} candidates):
Recall@1:  {100/num_queries:.2f}%
Recall@5:  {500/num_queries:.2f}%
Recall@10: {1000/num_queries:.2f}%
MRR:       {(np.log(num_queries) + 0.5772) / num_queries:.4f}
"""
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)

    print(summary)
    print(f"Charts saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()