import numpy as np
from scipy.stats import chi2

def mcnemar_test(correct_a, correct_b, name_a, name_b):
    # n01 = correct under A but wrong under B; n10 = wrong under A but correct under B
    n01 = np.sum((correct_a == 1) & (correct_b == 0))
    n10 = np.sum((correct_a == 0) & (correct_b == 1))

    if n01 + n10 == 0:
        print(f"{name_a} vs {name_b}: no discordant pairs, test undefined")
        return

    statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
    p_value = 1 - chi2.cdf(statistic, df=1)

    print(f"{name_a} vs {name_b}:")
    print(f"  n01 ({name_a} right, {name_b} wrong): {n01}")
    print(f"  n10 ({name_a} wrong, {name_b} right): {n10}")
    print(f"  McNemar statistic: {statistic:.3f}  (p-value: {p_value:.4f})")
    if p_value < 0.05:
        print(f"  → Statistically significant difference (p < 0.05)")
    else:
        print(f"  → NOT statistically significant (p >= 0.05)")
    print()

zeroshot = np.load('kaggle_outputs/zeroshot_correctness.npy')
fulltune = np.load('kaggle_outputs/fulltune_correctness.npy')
peft = np.load('kaggle_outputs/peft_correctness.npy')

print(f"Zero-shot correct: {zeroshot.sum()}/1000")
print(f"Full fine-tune correct: {fulltune.sum()}/1000")
print(f"PEFT correct: {peft.sum()}/1000\n")

mcnemar_test(fulltune, peft, "Full Fine-Tune", "PEFT")
mcnemar_test(zeroshot, fulltune, "Zero-Shot", "Full Fine-Tune")
mcnemar_test(zeroshot, peft, "Zero-Shot", "PEFT")