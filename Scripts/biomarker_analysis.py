import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
plt.bar(x - 0.2, top_data["Healthy"], 0.4)
plt.bar(x + 0.2, top_data["Disease"], 0.4)
plt.xticks(x, top_data["Gene"], rotation=90)
plt.xlabel("Top 20 Genes")
plt.ylabel("Expression Level")
plt.title("Top Differentially Expressed Genes")
plt.tight_layout()
plt.savefig("plots/gene_expression_top20.png")
plt.close()

p_values = []

for _, row in data.iterrows():
    healthy = [row["H1"], row["H2"], row["H3"]]
    disease = [row["D1"], row["D2"], row["D3"]]
    _, p_val = ttest_ind(healthy, disease, equal_var=False, nan_policy='omit')
    p_values.append(p_val)

data["p_value"] = p_values
data["adj_p_values"] = multipletests(data["p_value"], method="fdr_bh")[1]
data["neg_log10_adj_pval"] = -np.log10(data["adj_p_values"])

data["Regulation"] = "Not Significant"
data.loc[(data["log2FoldChange"] > 1) & (data["adj_p_values"] < 0.05), "Regulation"] = "Upregulated"
data.loc[(data["log2FoldChange"] < -1) & (data["adj_p_values"] < 0.05), "Regulation"] = "Downregulated"

colors = data["Regulation"].map({
    "Upregulated": "red",
    "Downregulated": "blue",
    "Not Significant": "grey"
})

plt.figure()
plt.scatter(data["log2FoldChange"], data["neg_log10_adj_pval"], c=colors)
plt.axvline(x=1, linestyle="--")
plt.axvline(x=-1, linestyle="--")
plt.axhline(y=-np.log10(0.05), linestyle="--")
plt.xlabel("log2(Fold Change)")
plt.ylabel("-log10(adjusted p-value)")
plt.title("Volcano Plot")
plt.tight_layout()
plt.savefig("plots/volcano_plot.png")
plt.close()

significant = data[
    (np.abs(data["log2FoldChange"]) > 1) &
    (data["adj_p_values"] < 0.05)
]

heatmap_data = significant[["H1", "H2", "H3", "D1", "D2", "D3"]]

plt.figure(figsize=(10, 6))
plt.imshow(heatmap_data, aspect='auto')
plt.colorbar()
plt.title("Heatmap of Significant Genes")
plt.xlabel("Samples")
plt.ylabel("Genes")
plt.tight_layout()
plt.savefig("plots/heatmap.png")
plt.close()

pca_data = data[["H1", "H2", "H3", "D1", "D2", "D3"]]

pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_data)

plt.figure()
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Plot")
plt.tight_layout()
plt.savefig("plots/pca.png")
plt.close()

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(pca_data)
data["Cluster"] = clusters

summary_counts = data["Regulation"].value_counts()

plt.figure()
summary_counts.plot(kind="bar")
plt.title("Gene Regulation Summary")
plt.tight_layout()
plt.savefig("plots/regulation_summary.png")
plt.close()

biomarkers = data[data["FoldChange"] > 2.5]

top_up = data[data["Regulation"] == "Upregulated"].head(10)
top_down = data[data["Regulation"] == "Downregulated"].head(10)

data.to_csv("results/full_results.csv", index=False)
biomarkers.to_csv("results/top_biomarkers.csv", index=False)
significant.to_csv("results/significant_genes.csv", index=False)
top_up.to_csv("results/top_upregulated.csv", index=False)
top_down.to_csv("results/top_downregulated.csv", index=False)

with open("results/analysis_summary.txt", "w") as f:
    f.write(f"Total genes: {len(data)}\n")
    f.write(f"Significant genes: {len(significant)}\n")
    f.write(f"Upregulated: {(data['Regulation']=='Upregulated').sum()}\n")
    f.write(f"Downregulated: {(data['Regulation']=='Downregulated').sum()}\n")

