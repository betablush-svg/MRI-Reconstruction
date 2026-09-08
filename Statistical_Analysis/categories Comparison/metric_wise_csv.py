import pandas as pd

models = {
    'Classical': pd.read_csv("CLASSICAL_metrics.csv"),
    'Early_DL': pd.read_csv("Early_DL_metrics.csv"),
    'Physics_DL': pd.read_csv("Physics_DL_metrics.csv"),
    'Adv_Gen_DL': pd.read_csv("Advanced_Gen_metrics.csv")
}

metrics = ["NMSE", "PSNR", "SSIM", "HFEN", "VIF", "LPIPS", "TIME"]

# Traverse over metrics and save CSV
for metric in metrics:
    # Collect metric values across models
    df_metric = pd.DataFrame()
    for model, df in models.items():
        df_metric[model] = df[metric]

    # Save to CSV
    filename = f"{metric}_category.csv"
    df_metric.to_csv(filename, index=False)
    print(f"Saved {filename}") 