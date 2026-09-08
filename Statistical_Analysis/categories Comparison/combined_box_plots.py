import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

metrics = ["NMSE", "PSNR", "SSIM", "HFEN", "VIF", "LPIPS", "TIME"]
selected_models = ['Classical', 'Early DL', 'Physics DL', 'Adv DL']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    df_metric = pd.read_csv(f"{metric}_category.csv")
    df_plot = df_metric[selected_models]
    df_long = df_plot.melt(var_name="Model Category", value_name=metric)

    ax = axes[i]

    sns.violinplot(x="Model Category", y=metric, data=df_long, inner=None, color=".85", ax=ax)
    sns.boxplot(x="Model Category", y=metric, data=df_long, width=0.25, ax=ax)

    ax.set_title(metric, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    if metric in ["NMSE", "HFEN", "LPIPS", "TIME"]:
        ax.invert_yaxis()

# Remove unused subplot
fig.delaxes(axes[-1])

# 🔧 IMPORTANT: spacing adjustments
plt.subplots_adjust(
    left=0.05,
    right=0.98,
    top=0.90,     # leave space for title
    bottom=0.12,
    hspace=0.55,  # vertical spacing
    wspace=0.30   # horizontal spacing
)

# Add super title AFTER spacing adjustment
fig.suptitle("Category Wise Comparison Across Metrics", fontsize=18)

plt.show()