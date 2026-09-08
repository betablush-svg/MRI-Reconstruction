import pandas as pd
import matplotlib.pyplot as plt
metrics = ["NMSE", "PSNR", "SSIM", "HFEN", "VIF", "LPIPS", "TIME"]
models = ['Classical', 'Early_DL', 'Physics_DL', 'Adv_Gen_DL']

# Load data
normalized = {}

for metric in metrics:
    df = pd.read_csv(f"{metric}_category.csv")
    vals = df[models].mean()

    if metric in ["PSNR", "SSIM", "VIF"]:  # higher is better
        norm = (vals - vals.min()) / (vals.max() - vals.min())
    else:  # lower is better
        norm = (vals.max() - vals) / (vals.max() - vals.min())

    normalized[metric] = norm

df_norm = pd.DataFrame(normalized)
df_time = pd.read_csv("TIME_category.csv")

weights = {
    "PSNR": 0.2,
    "SSIM": 0.3,
    "VIF": 0.2,
    "LPIPS": 0.3,
    "NMSE": 0.2,
    "HFEN": 0.1,
    "TIME": 0.1
}
# Compute means
score = sum(df_norm[m] * w for m, w in weights.items()) / sum(weights.values())
time_mean = df_time[models].mean()

colors = {
    "Classical": "#4C72B0",
    "Early_DL": "#55A868",
    "Physics_DL": "#C44E52",
    "Adv_Gen_DL": "#8172B2"
}

plt.figure(figsize=(7,6))

for model in models:
    plt.scatter(
        time_mean[model],
        score[model],
        s=120,
        color=colors[model],
        label=model
    )

    plt.text(
        time_mean[model] + 0.0002,
        score[model],
        model,
        fontsize=10
    )

plt.xlabel("Inference Time (seconds)")
plt.ylabel("Quality (Weighted Normalized Metrics)")
plt.title("Time vs Quality Trade-off")

plt.grid(alpha=0.3)
plt.legend()

x_min, x_max = time_mean.min(), time_mean.max()
y_min, y_max = score.min(), score.max()

dx = (x_max - x_min) * 0.1
dy = (y_max - y_min) * 0.1

# Better (top-left direction)
plt.annotate(
    "Better ↑",
    xy=(x_min, y_max),
    xytext=(x_min + dx, y_max - dy),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10
)

# Faster (left direction)
plt.annotate(
    "Faster ←",
    xy=(x_min, y_min),
    xytext=(x_min + dx, y_min + dy),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10
)

plt.tight_layout()

points = list(zip(time_mean, score))

# Sort by time
points_sorted = sorted(points)

pareto = []
best_psnr = -1

for t, p in points_sorted:
    if p > best_psnr:
        pareto.append((t, p))
        best_psnr = p

# Plot line
px, py = zip(*pareto)
plt.plot(px, py, linestyle='--', color='black')
plt.show()