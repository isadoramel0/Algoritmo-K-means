import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import time
import tracemalloc

#  Carregar os dados 
iris = load_iris()
X = iris.data
y_true = iris.target

# Normalizar os dados: cada atributo passa a ter média 0 e desvio padrão 1
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

for k in [3, 5]:
    print(f"\n K-means para k={k}:")

    # Medir tempo e memória 
    tracemalloc.start()
    inicio = time.time()

    # Treina o modelo, obtém os rótulos preditos para cada ponto e os centróides finais
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)
    centroids = kmeans.cluster_centers_

    fim = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Métricas de avaliação 
    inertia = kmeans.inertia_
    sil = silhouette_score(Xs, labels) if k > 1 else float("nan")
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    print(f"  k = {k}, silhouette = {sil:.4f}, inertia = {inertia:.4f}")
    print(f"  ARI = {ari:.4f}, NMI = {nmi:.4f}")
    print(f"  Tempo de execucao: {fim - inicio:.4f} segundos")
    print(f"  Memoria usada: atual {current/1024:.2f} KB, pico {peak/1024:.2f} KB")

    # Comparação clusters x classes verdadeiras
    df = pd.DataFrame({'true': y_true, 'cluster': labels})
    cross = pd.crosstab(df['cluster'], df['true'])
    print("\nComparacao dos clusters com as classes verdadeiras:")
    print(f"\nk = {k}\n{cross}")

    # Redução de dimensionalidade PCA 
    for n_comp in [1, 2]:
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(Xs)
        centroids_pca = pca.transform(centroids)

        plt.figure(figsize=(6,5))
        if n_comp == 1:
            # Gráfico 1D (scatter em linha)
            for cluster in range(k):
                pts = X_pca[labels == cluster]
                plt.scatter(pts, np.zeros_like(pts), label=f'Cluster {cluster}', alpha=0.7)
            plt.scatter(centroids_pca, np.zeros_like(centroids_pca), marker='X', s=200, c='black', label='Centroids')
            plt.xlabel("PCA 1")
            plt.title(f"KMeans (k={k}) com PCA(1)")
            plt.legend()
        else:
            # Gráfico 2D
            for cluster in range(k):
                pts = X_pca[labels == cluster]
                plt.scatter(pts[:,0], pts[:,1], label=f'Cluster {cluster}', alpha=0.7)
            plt.scatter(centroids_pca[:,0], centroids_pca[:,1], marker='X', s=200, c='black', label='Centroids')
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.title(f"KMeans (k={k}) com PCA(2)")
            plt.legend()

        plt.show()
