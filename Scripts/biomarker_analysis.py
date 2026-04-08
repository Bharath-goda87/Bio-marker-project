import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

file_path = "data/gene_expression.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

data = pd.read_csv(file_path)

required_cols = ["Healthy", "Disease", "H1", "H2", "H3", "D1", "D2", "D3"]
data = data.dropna(subset=required_cols)
data = data.fillna(data.select_dtypes(include=[np.number]).mean())

data["Healthy"] = data["Healthy"].replace(0, np.nan)
data["FoldChange"] = data["Disease"] / data["Healthy"]
data["log2FoldChange"] = np.log2(data["FoldChange"])

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)


os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)


top_data = data.sort_values(by="FoldChange", ascending=False).head(20)

plt.figure(figsize=(14, 6))
x = np.arange(len(top_data))

plt.bar(x - 0.2, top_data["Healthy"], 0.4, label="Healthy")
plt.bar(x + 0.2, top_data["Disease"], 0.4, label="Disease")

plt.xticks(x, top_data["Gene"], rotation=90)
plt.xlabel("Top 20 Genes")
plt.ylabel("Expression Level")
plt.title("Top Differentially Expressed Genes")
plt.legend()
plt.tight_layout()
plt.savefig("plots/gene_expression_top20.png")
plt.close()

p_values = []

for _, row in data.iterrows():
    healthy = [row["H1"], row["H2"], row["H3"]]
    disease = [row["D1"], row["D2"], row["D3"]]

    stat, p_val = ttest_ind(healthy, disease, equal_var=False, nan_policy='omit')
    p_values.append(p_val)

data["p_value"] = p_values


data["adj_p_values"] = multipletests(data["p_value"], method="fdr_bh")[1]


data["neg_log10_adj_pval"] = -np.log10(data["adj_p_values"])

plt.figure(figsize=(8, 6))
plt.scatter(data["log2FoldChange"], data["neg_log10_adj_pval"])

plt.axvline(x=1, linestyle="--")
plt.axvline(x=-1, linestyle="--")
plt.axhline(y=-np.log10(0.05), linestyle="--")

plt.xlabel("log2(Fold Change)")
plt.ylabel("-log10(adjusted p-value)")
plt.title("Volcano Plot (Adjusted p-values)")
plt.tight_layout()
plt.savefig("plots/volcano_plot.png")
plt.close()

significant = data[
    (np.abs(data["log2FoldChange"]) > 1) &
    (data["adj_p_values"] < 0.05)
]

heatmap_data = significant[["H1","H2","H3","D1","D2","D3"]]

plt.figure(figsize=(10, 6))
plt.imshow(heatmap_data, aspect='auto')
plt.colorbar()
plt.title("Heatmap of Significant Genes")
plt.xlabel("Samples")
plt.ylabel("Genes")
plt.tight_layout()
plt.savefig("plots/heatmap.png")
plt.close()

biomarkers = data[data["FoldChange"] > 2.5]

data.to_csv("results/full_results.csv", index=False)
biomarkers.to_csv("results/top_biomarkers.csv", index=False)
significant.to_csv("results/significant_genes.csv", index=False)

print("\nSummary:")
print(f"Total genes: {len(data)}")
print(f"Significant genes: {len(significant)}")
print(f"Upregulated: {(data['log2FoldChange'] > 1).sum()}")
print(f"Downregulated: {(data['log2FoldChange'] < -1).sum()}")

print("\nTop Biomarkers:")
print(
    biomarkers[["Gene", "FoldChange"]]
    .sort_values(by="FoldChange", ascending=False)
    .head()
)

print("\nTop Significant Genes:")
print(
    significant[["Gene", "log2FoldChange", "adj_p_values"]]
    .sort_values(by="adj_p_values")
    .head()
)