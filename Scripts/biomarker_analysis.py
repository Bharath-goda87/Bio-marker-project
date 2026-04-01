import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

data = pd.read_csv("../data/gene_expression.csv")

data = data.dropna(subset=["Healthy", "Disease", "H1", "H2", "H3", "D1", "D2", "D3"])
data.fillna(data.mean(numeric_only=True), inplace=True)

data["FoldChange"] = data["Disease"] / data["Healthy"]
data["log2FoldChange"] = np.log2(data["FoldChange"].replace(0, np.nan))

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(20, 5))
x = np.arange(len(data["Gene"]))
width = 0.4

plt.bar(x - width/2, data["Healthy"], width, label="Healthy")
plt.bar(x + width/2, data["Disease"], width, label="Disease")

plt.xticks(x, data["Gene"], rotation=90)
plt.xlabel("Genes")
plt.ylabel("Expression Level")
plt.title("Gene Expression: Healthy vs Disease")
plt.legend()
plt.tight_layout()
plt.savefig("plots/gene_expression.png")
plt.close()

p_values = []

for _, row in data.iterrows():
    healthy = [row["H1"], row["H2"], row["H3"]]
    disease = [row["D1"], row["D2"], row["D3"]]
    
    stat, p_val = ttest_ind(healthy, disease, equal_var=False)
    p_values.append(p_val)

data["p_value"] = p_values

data["adj_p_values"] = multipletests(data["p_value"], method="fdr_bh")[1]

data["neg_log10_pval"] = -np.log10(data["p_value"])

plt.figure(figsize=(8, 6))
plt.scatter(data["log2FoldChange"], data["neg_log10_pval"], alpha=0.7)

plt.axvline(x=-1, linestyle="--")
plt.axvline(x=1, linestyle="--")
plt.axhline(y=-np.log10(0.05), linestyle="--")

plt.xlabel("log2(Fold Change)")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot")
plt.tight_layout()
plt.savefig("plots/volcano_plot.png")
plt.close()

biomarkers = data[data["FoldChange"] > 2.5]

significant = data[
    (np.abs(data["log2FoldChange"]) > 1) &
    (data["adj_p_values"] < 0.05)
]

print("Top Biomarkers:")
print(biomarkers[["Gene", "FoldChange"]].sort_values(by="FoldChange", ascending=False).head())

print("\nSignificant Genes:")
print(significant[["Gene", "log2FoldChange", "adj_p_values"]].sort_values(by="adj_p_values"))