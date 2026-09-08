import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metrics = ["PSNR", "SSIM", "VIF", "NMSE", "HFEN", "LPIPS", "TIME"]
selected_models = ['Classical', 'Early DL', 'Physics DL', 'Adv DL']

# Combine mean values
data = {}

for metric in metrics:
    df = pd.read_csv(f"{metric}_category.csv")
    data[metric] = df[selected_models].mean()

df_all = pd.DataFrame(data)
print(df_all.head())
# ----------------------------
# Normalize (0–1 scale)
# ----------------------------
df_norm = df_all.copy()


for metric in metrics:
    vals = df_all[metric]

    if metric in ["PSNR", "SSIM", "VIF"]:  # higher is better
        df_norm[metric] = (vals - vals.min()) / (vals.max() - vals.min())
    else:  # lower is better
        df_norm[metric] = (vals.max() - vals) / (vals.max() - vals.min())
print(df_norm.head())
# ----------------------------
# Radar Plot
# ----------------------------
labels = metrics
num_vars = len(labels)

angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])  # close loop

fig = plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)

for model in selected_models:
    values = df_norm.loc[model].values
    values = np.concatenate([values, [values[0]]])

    ax.plot(angles, values, label=model)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.set_title("Multi-Metric Comparison (Normalized)", fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()