import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from scipy.stats import rankdata
from scipy.stats import wilcoxon

import numpy as np

metrics = ["NMSE", "PSNR", "SSIM", "HFEN", "VIF", "LPIPS", "TIME"]

#friedman test for statistical significance

# comparing 3 models - UNET, Dual Domain, ADMM

# Hypothesis - All models perform equally
# Alternate hypothesis - at least one model performs differently
statistics = []
p_values = []

wilcoxon_stat = {'Model1VsModel2': [
    'Classical vs Early_DL', 
    'Classical vs Physics_DL',  
    'Classical vs Adv_Gen_DL',
    'Early_DL vs Physics_DL',
    'Early_DL vs Adv_Gen_DL',
    'Physics_DL vs Adv_Gen_DL',
    'friedmansquare_stat',
]}

wilcoxon_p = {'Model1VsModel2': [
    'Classical vs Early_DL', 
    'Classical vs Physics_DL',  
    'Classical vs Adv_Gen_DL',
    'Early_DL vs Physics_DL',
    'Early_DL vs Adv_Gen_DL',
    'Physics_DL vs Adv_Gen_DL',
    'friedmansquare_p'
]}

mean_d = {'Model1-Model2': [
    'Classical - Early_DL', 
    'Classical - Physics_DL',  
    'Classical - Adv_Gen_DL',
    'Early_DL - Physics_DL',
    'Early_DL - Adv_Gen_DL',
    'Physics_DL - Adv_Gen_DL'
]}

median_d = {'Model1-Model2': [
    'Classical - Early_DL', 
    'Classical - Physics_DL',  
    'Classical - Adv_Gen_DL',
    'Early_DL - Physics_DL',
    'Early_DL - Adv_Gen_DL',
    'Physics_DL - Adv_Gen_DL'
]}

# PSNR metric
for metric in metrics:
    df_metric = pd.read_csv(f"{metric}_category.csv")
    models = ['Classical', 'Early_DL', 'Physics_DL', 'Adv_Gen_DL']
    print(f"Evaluating {metric} Metric: ")

    data = np.array([
        df_metric['Classical'].values,
        df_metric['Early_DL'].values,
        df_metric['Physics_DL'].values,
        df_metric['Adv_Gen_DL'].values
    ]).T

    stat, p = friedmanchisquare(df_metric['Classical'].values,df_metric['Early_DL'].values,df_metric['Physics_DL'].values, df_metric['Adv_Gen_DL'].values)
    statistics.append(stat)
    p_values.append(p)

    # print(df_psnr.head())
    print("Statistics: ", stat)
    print("P Value: ", p)

    if(p<0.05):
        print("Significance Difference Exist")
    else:
        print("No Significant difference")

    #data = np.array([unet, dual_domain, admm]).T

    print("Shape: ", data.shape)
    print("Classical vs Early_DL:", wilcoxon(data[:,0], data[:,1]))
    print("Classical vs Physics_DL:", wilcoxon(data[:,0], data[:,2]))
    print("Classical vs Adv_Gen_DL:", wilcoxon(data[:,0], data[:,3]))
    print("Early_DL vs Physics_DL:", wilcoxon(data[:,1], data[:,2]))
    print("Early_DL vs Adv_Gen_DL:", wilcoxon(data[:,1], data[:,3]))
    print("Physics_DL vs Adv_Gen_DL:", wilcoxon(data[:,2], data[:,3]))

    wilcoxon_stat[metric] = [
        wilcoxon(data[:,0], data[:,1])[0],
        wilcoxon(data[:,0], data[:,2])[0],
        wilcoxon(data[:,0], data[:,3])[0],
        wilcoxon(data[:,1], data[:,2])[0],
        wilcoxon(data[:,1], data[:,3])[0],
        wilcoxon(data[:,2], data[:,3])[0],
        stat
    ]

    wilcoxon_p[metric] = [
        wilcoxon(data[:,0], data[:,1])[1],
        wilcoxon(data[:,0], data[:,2])[1],
        wilcoxon(data[:,0], data[:,3])[1],
        wilcoxon(data[:,1], data[:,2])[1],
        wilcoxon(data[:,1], data[:,3])[1],
        wilcoxon(data[:,2], data[:,3])[1],
        p
    ]


    print("Classical - Early_DL:", np.mean(data[:,0] - data[:,1]))
    print("Classical - Physics_DL:", np.mean(data[:,0] - data[:,2]))
    print("Classical - Adv_Gen_DL:", np.mean(data[:,0] - data[:,3]))
    print("Early_DL - Physics_DL:", np.mean(data[:,1] - data[:,2]))
    print("Early_DL - Adv_Gen_DL:", np.mean(data[:,1] - data[:,3]))
    print("Physics_DL - Adv_Gen_DL:", np.mean(data[:,2] - data[:,3]))

    mean_d[metric] = [
        np.mean(data[:,0] - data[:,1]),
        np.mean(data[:,0] - data[:,2]),
        np.mean(data[:,0] - data[:,3]),
        np.mean(data[:,1] - data[:,2]),
        np.mean(data[:,1] - data[:,3]),
        np.mean(data[:,2] - data[:,3])
    ]

    print("Median Classical - Early_DL:", np.median(data[:,0] - data[:,1]))
    print("Median Classical - Physics_DL:", np.median(data[:,0] - data[:,2]))
    print("Median Classical - Adv_Gen_DL:", np.median(data[:,0] - data[:,3]))
    print("Median Early_DL - Physics_DL:", np.median(data[:,1] - data[:,2]))
    print("Median Early_DL - Adv_Gen_DL:", np.median(data[:,1] - data[:,3]))
    print("Median Physics_DL - Adv_Gen_DL:", np.median(data[:,2] - data[:,3]))

    median_d[metric] = [
        np.median(data[:,0] - data[:,1]),
        np.median(data[:,0] - data[:,2]),
        np.median(data[:,0] - data[:,3]),
        np.median(data[:,1] - data[:,2]),
        np.median(data[:,1] - data[:,3]),
        np.median(data[:,2] - data[:,3])
    ]

wilc_stat_df = pd.DataFrame(wilcoxon_stat)
wilc_p_df = pd.DataFrame(wilcoxon_p)
print(wilc_stat_df)
mean_df = pd.DataFrame(mean_d)
median_df = pd.DataFrame(median_d)

wilc_stat_df.to_csv('Wilcoxon-stat-4-models.csv')
wilc_p_df.to_csv('Wilcoxon-p-4-models.csv')
mean_df.to_csv('Mean-4-models.csv')
median_df.to_csv('Median-4-models.csv')

print("Saved to CSV")
#nemenyi = sp.posthoc_nemenyi(data)

#print(data.shape)
#print(data[:5])
#print(nemenyi)