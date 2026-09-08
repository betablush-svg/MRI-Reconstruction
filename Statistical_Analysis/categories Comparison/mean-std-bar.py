import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

metrics = ["NMSE", "PSNR", "SSIM", "HFEN", "VIF", "LPIPS", "TIME"]

for metric in metrics:
    df_metric = pd.read_csv(f"{metric}_category.csv")
    means = df_metric.mean()
    stds = df_metric.std()

    means.plot(kind='bar', yerr=stds)
    plt.xticks(rotation=45, ha='right')  # clearer angle
    plt.subplots_adjust(bottom=0.25)     # more space for labels

    plt.title(f"{metric} Mean ± Std")
    plt.show()