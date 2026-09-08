import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import scikit_posthocs as sp
from aeon.visualisation import plot_critical_difference
metrics = ["NMSE", "PSNR", "SSIM", "HFEN", "VIF", "LPIPS", "TIME"]
selected_models = ['Classical', 'Early_DL', 'Physics_DL', 'Adv_Gen_DL']

# stack ranks across metrics
all_ranks = []

for metric in metrics:
    df = pd.read_csv(f"{metric}_category.csv")
    data = df[selected_models].values

    if metric in ["PSNR", "SSIM", "VIF"]:
        ranks = np.array([rankdata(-row) for row in data])
    else:
        ranks = np.array([rankdata(row) for row in data])

    all_ranks.append(ranks)


all_ranks = np.concatenate(all_ranks, axis=0)

avg_ranks = all_ranks.mean(axis=0)
print(all_ranks.shape)
plot_critical_difference(all_ranks, selected_models, test="wilcoxon")
plt.show()