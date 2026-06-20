import numpy as np
import pandas as pd

image_embeds = np.load('kaggle_outputs/val_image_embeddings_v2.npy')
text_embeds = np.load('kaggle_outputs/val_text_embeddings_v2.npy')
captions = pd.read_csv('kaggle_outputs/val_captions.csv')
error_log = pd.read_csv('kaggle_outputs/medclip_error_log.csv')

# 1. Modality gap — how far apart image embeddings and text embeddings sit on average.
# Real phenomenon documented in CLIP literature (Liang et al., NeurIPS 2022) — not unique to this model.
img_centroid = image_embeds.mean(axis=0)
txt_centroid = text_embeds.mean(axis=0)
gap = np.linalg.norm(img_centroid - txt_centroid)
print(f"Modality gap (centroid distance): {gap:.4f}")
print("  (max possible distance between unit vectors is 2.0)")

# 2. Build a rank column — rank 1 means correct top-1 match (not in error_log),
# everything else comes straight from the saved error log.
ranks = pd.Series(1, index=range(len(captions)))
ranks.loc[error_log['query_index']] = error_log['actual_rank'].values
df = captions.copy()
df['rank'] = ranks.values
df['caption_length'] = df['caption'].str.split().str.len()

# 3. Does caption length relate to how well it gets matched?
corr = df['caption_length'].corr(df['rank'])
print(f"\nCorrelation between caption length and rank: {corr:.3f}")
print("  (negative = longer/more detailed captions tend to rank better)")

# 4. How many captions are exact duplicates of each other in this 1000-pair pool?
# High duplicate counts mean Recall@1 is partly capped by the dataset itself,
# not just the model — several images can share a near-identical correct caption.
dup_counts = df['caption'].value_counts()
exact_dupes = (dup_counts > 1).sum()
print(f"\nDuplicate captions in this 1000-pair pool: {dup_counts[dup_counts > 1].sum()} "
      f"(across {exact_dupes} duplicate groups)")

# 5. Performance broken down by scan type, via simple keyword matching on the caption text.
modality_keywords = {
    'CT': r'\bCT\b|computed tomography',
    'MRI': r'\bMRI\b|magnetic resonance',
    'X-ray': r'\bx-?ray\b|radiograph',
    'Ultrasound': r'ultrasound|sonograph',
}
print("\nPer-modality average rank (lower number = better):")
for label, pattern in modality_keywords.items():
    mask = df['caption'].str.contains(pattern, case=False, regex=True, na=False)
    if mask.sum() > 0:
        print(f"  {label:<12} n={mask.sum():<5} avg rank={df.loc[mask, 'rank'].mean():.1f}  median={df.loc[mask,'rank'].median():.0f}")

df.to_csv('failure_mode_breakdown.csv', index=False)
print("\nSaved failure_mode_breakdown.csv")