import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statannotations.Annotator import Annotator

metrics = ["NMSE", "PSNR", "SSIM", "HFEN", "VIF", "LPIPS", "TIME"]
selected_models = ['UNET', 'DUAL_DOMAIN', 'ADMM', 'ZFRecon']
for metric in metrics:
    df_metric = pd.read_csv(f"{metric}_comparison_x4.csv")
    
    # Keep only selected models
    df_plot = df_metric[selected_models]

    # Convert to long format (important for seaborn)
    df_long = df_plot.melt(var_name="Model", value_name=metric)

    plt.figure(figsize=(8,5))
    sns.violinplot(x="Model", y=metric, data=df_long, inner=None, color='.85')


    sns.boxplot(
        x="Model",
        y=metric,
        data=df_long,
        width=0.2
    )
    sns.stripplot(x="Model", y=metric, data=df_long, color='black', alpha=0.15, size=1)
    
    plt.title(f"{metric} Distribution Across Models")
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.show()