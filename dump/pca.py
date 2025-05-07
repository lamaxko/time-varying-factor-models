import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Run PCA to reduce 5D h_t states to 2D for visualization
pca = PCA(n_components=2)
h_t_pca = pca.fit_transform(h_t_df.values)

# Create a DataFrame for plotting
h_t_pca_df = pd.DataFrame(h_t_pca, columns=["PC1", "PC2"], index=h_t_df.index)

# Plot the PCA-transformed latent macro states
plt.figure(figsize=(10, 6))
sns.scatterplot(data=h_t_pca_df, x="PC1", y="PC2", hue=h_t_pca_df.index.year, palette="viridis", legend=False)
plt.title("PCA of Latent Macroeconomic States (hₜ)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
