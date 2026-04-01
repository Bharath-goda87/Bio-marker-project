import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as  np
from scipy.stats import test_ind
from statsmodels.stats.multitest import multipletests
data = pd.read_csv("../data/gene_expression.csv")
print(data)
data["FoldChange"] = data["Disease"] / data["Healthy"]
print("\nFold Change Results:")
print(data)
os.makedirs("plots",exist_ok=True)
plt.figure(figsize=(20,5))
plt.bar(data["Gene"], data["Healthy"], label="Healthy", alpha=0.7)
plt.bar(data["Gene"], data["Disease"], label="Disease", alpha=0.7)
plt.xlabel("Genes")
plt.ylabel("Gene Expression Level")
plt.title("Gene Expression: Healthy vs Disease")
plt.legend()
plt.savefig("plots/gene_expression.png")
plt.show()
print(data.head())
print(data.info())
print(data.isnull().sum())
data.fillna(data.mean(numeric_only=True),inplace=True)
print("Data is clean and normalized.")
biomarker = data[data["FoldChange"] > 2.5]
print("potential biomarker")
print(biomarker)
shortdata = data.sort_values(by="FoldChange",ascending=True);
print(shortdata)
data["log2FoldChange"] = np.log2(data["FoldChange"])
np.random.seed(0)
data["p_value"] = np.random.uniform(low=0.001,high=1.0, size=len(data))
data["neg_log10_pval"] = -np.log10(data["p_value"])
plt.figure(figsize=(8,6))
plt.scatter(data["log2FoldChange"], data["neg_log10_pval"],c="grey")
plt.axvline(x=-1, color="red", linestyle="--")
plt.axvline(x=1, color="red", linestyle="--")
plt.axhline(y=np.log10(0.05), color="blue", linestyle="--")
plt.xlabel("log2(Fold Change)")
plt.ylabel("-log10(p-value)")
plt.title("volcano plot of Gene Expression")
plt.show()
np.random.seed(0)
data["p_value"] = np.random.uniform(...)
singification = data[(abs(data['log2FoldChange']) > 1) & (data["p_value"] < 0.05)]
list_values = []
for index, rows in data.iterrows():
    healthy =[rows["H1"], rows["H2"], rows["H3"]]
    disease  =[rows["D1"], rows["D2"],rows["D3"]]
state,d =  test_ind(healthy,disease)
list_values.append(d)
data["p_value"] = list_values
data['adj_p_values'] = multipletests(data['p_value'],method='far_dh')[1]
singification = data[
(abs(data['log2FoldChange'])>1) & data(['adj_p_values'] < 0.05)    
]
print(singification)