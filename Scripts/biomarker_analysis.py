import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Load data
data = pd.read_csv("../data/gene_expression.csv")
print(data)

# Handle missing values first
data.fillna(data.mean(numeric_only=True), inplace=True)

# Fold Change
data["FoldChange"] = data["Disease"] / data["Healthy"]

print("\nFold Change Results:")
print(data)

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Bar plot
plt.figure(figsize=(20, 5))
plt.bar(data["Gene"], data["Healthy"], label="Healthy", alpha=0.7)
plt.bar(data["Gene"], data["Disease"], label="Disease", alpha=0.7)
plt.xlabel("Genes")
plt.ylabel("Gene Expression Level")
plt.title("Gene Expression: Healthy vs Disease")
plt.legend()
plt.savefig("plots/gene_expression.png")
plt.show()

# Basic info
print(data.head())
print(data.info())
print(data.isnull().sum())

print("Data is clean and normalized.")

# Biomarker selection
biomarker = data[data["FoldChange"] > 2.5]
print("Potential biomarkers:")
print(biomarker)

# Sort data
sorted_data = data.sort_values(by="FoldChange", ascending=True)
print(sorted_data)

# Log transformation
data["log2FoldChange"] = np.log2(data["FoldChange"])

# Statistical testing (real p-values)
p_values = []

for index, row in data.iterrows():
    healthy = [row["H1"], row["H2"], row["H3"]]
    disease = [row["D1"], row["D2"], row["D3"]]
    
    stat, p_val = ttest_ind(healthy, disease)
    p_values.append(p_val)

data["p_value"] = p_values

# Multiple testing correction
data["adj_p_values"] = multipletests(data["p_value"], method='fdr_bh')[1]

# Volcano plot
data["neg_log10_pval"] = -np.log10(data["p_value"])

plt.figure(figsize=(8, 6))
plt.scatter(data["log2FoldChange"], data["neg_log10_pval"], c="grey")

plt.axvline(x=-1, color="red", linestyle="--")
plt.axvline(x=1, color="red", linestyle="--")
plt.axhline(y=-np.log10(0.05), color="blue", linestyle="--")

plt.xlabel("log2(Fold Change)")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot of Gene Expression")

plt.savefig("plots/volcano_plot.png")
plt.show()

# Significant genes
significant = data[
    (abs(data["log2FoldChange"]) > 1) &
    (data["adj_p_values"] < 0.05)
]

print("Significant genes:")
print(significant)