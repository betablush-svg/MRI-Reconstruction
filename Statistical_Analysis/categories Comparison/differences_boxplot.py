import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

metric = "PSNR"
df_metric = pd.read_csv(f"{metric}_category.csv")

diff_df = pd.DataFrame({
    "Classical-Early_DL": df_metric['Classical']-df_metric['Early_DL'],
    "Classical-Physics_DL":df_metric['Classical']-df_metric['Physics_DL'],
    "Classical-Adv_Gen_DL":df_metric['Classical']-df_metric['Adv_Gen_DL'],
    "Early_DL-Physics_DL":df_metric['Early_DL']-df_metric['Physics_DL'],
    "Early_DL-Adv_Gen_DL":df_metric['Early_DL']-df_metric['Adv_Gen_DL'],
    "Physics_DL-Adv_Gen_DL":df_metric['Physics_DL']-df_metric['Adv_Gen_DL'],
})

sns.boxplot(data=diff_df)
plt.axhline(0, color='red')
plt.show()

means = df_metric.mean()
stds = df_metric.std()

means.plot(kind='bar', yerr=stds)
plt.title("PSNR Mean ± Std")
plt.show()