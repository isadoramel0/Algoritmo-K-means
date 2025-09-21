import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import os
import time
import tracemalloc 

# Inicializa centróides escolhendo k pontos aleatórios do dataset
def inicializar_centroides(X, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(X.shape[0], size=k, replace=False)
    return X[indices].astype(float)

# Atribui cada ponto ao cluster mais próximo (menor distância euclidiana)
def atribuir_clusters(X, centroides):
    distanciaEuclidiana = np.sum((X[:, np.newaxis, :] - centroides[np.newaxis, :, :])**2, axis=2)
    return np.argmin(distanciaEuclidiana, axis=1)

# Atualiza centróides calculando a média dos pontos em cada cluster
def atualizar_centroides(X, labels, k):
    numCaracteristicas = X.shape[1]
    novosCentroides = np.zeros((k, numCaracteristicas))
    for i in range(k):
        pontos = X[labels == i]
        if pontos.shape[0] == 0:
            novosCentroides[i] = X[np.random.randint(0, X.shape[0])]
        else:
            novosCentroides[i] = pontos.mean(axis=0)
    return novosCentroides

# Executa o algoritmo K-means até convergência ou máximo de iterações
def kmeans(X, k, max_iteracoes=300, tolerancia=1e-4, seed=None, verbose=False):
    centroides = inicializar_centroides(X, k, seed)
    for iteracao in range(max_iteracoes):
        labels = atribuir_clusters(X, centroides)
        novosCentroides = atualizar_centroides(X, labels, k)
        deslocamento = np.linalg.norm(novosCentroides - centroides, axis=1).max()
        centroides = novosCentroides
        if verbose:
            print(f"Iter {iteracao:03d}: max deslocamento centroides = {deslocamento:.6f}") 
        if deslocamento <= tolerancia:
            break
    return labels, centroides

# Roda o fluxo completo: K-means, métricas, PCA e salvamento do gráfico
def rodar_fluxo(k, X, y_true=None, seed=42, save_dir="resultados", verbose=False):
    os.makedirs(save_dir, exist_ok=True)
    melhorSolucao = None
    melhorInertia = np.inf
    numInicializacoes = 10

    for init in range(numInicializacoes):
        s = seed + init
        labels, centroides = kmeans(X, k, seed=s, verbose=False)
        distanciaEuclidiana = np.sum((X - centroides[labels])**2)
        if distanciaEuclidiana < melhorInertia:
            melhorInertia = distanciaEuclidiana
            melhorSolucao = (labels.copy(), centroides.copy(), s)

    labels, centroides, used_seed = melhorSolucao
    sil = silhouette_score(X, labels) if k > 1 else float('nan')

    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
    else:
        ari = float('nan')
        nmi = float('nan')

    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(6,5))
    for cluster in range(k):
        pts = X2[labels == cluster]
        plt.scatter(pts[:,0], pts[:,1], label=f'Cluster {cluster}', alpha=0.7, s=40)
    centroids2 = pca.transform(centroides)
    plt.scatter(centroids2[:,0], centroids2[:,1], marker='X', s=120, c='black', label='Centroids')
    plt.title(f'K-means (k={k}) - Silhouette: {sil:.4f}')
    plt.legend()
    filename = os.path.join(save_dir, f'kmeans_k{k}_seed{used_seed}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'k': k,
        'labels': labels,
        'centroids': centroides,
        'silhouette': sil,
        'inertia': melhorInertia,
        'ARI': ari,
        'NMI': nmi,
        'seed_used': used_seed,
        'plot_file': filename
    }

# Função principal: carrega dados, executa K-means para k escolhidos e mostra resultados
def main():
    iris = load_iris()
    X = iris.data
    y_true = iris.target

    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)

    resultados = []
    for k in [3, 5]:
        print(f"\nProcessando algoritmo K-means para k={k} ...")
        tracemalloc.start()
        inicio = time.time()
        res = rodar_fluxo(k, Xs, y_true=y_true, seed=42, save_dir="resultados", verbose=False)
        fim = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        resultados.append(res)
        print(f"  k = {k}, silhouette = {res['silhouette']:.4f}, inertia = {res['inertia']:.4f}")
        print(f"  ARI = {res['ARI']:.4f}, NMI = {res['NMI']:.4f}")
        print(f"  Figura salva em: {res['plot_file']}")
        print(f"  Tempo de execução: {fim - inicio:.4f} segundos")
        print(f"  Memória usada: atual {current/1024:.2f} KB, pico {peak/1024:.2f} KB")

    print("\nResumo comparativo:")
    for res in resultados:
        print(f"  k={res['k']}  -> silhouette={res['silhouette']:.4f}, inertia={res['inertia']:.4f}, "
              f"ARI={res['ARI']:.4f}, NMI={res['NMI']:.4f}, seed={res['seed_used']}")

    print("\nComparação dos clusters com as classes verdadeiras (apenas análise):")
    for res in resultados:
        df = pd.DataFrame({'true': y_true, 'cluster': res['labels']})
        cross = pd.crosstab(df['cluster'], df['true'])
        print(f"\nk = {res['k']}\n{cross}")

    print("\nFim. Verifique as figuras na pasta 'resultados' e as métricas impressas para cada k.")

if __name__ == "__main__":
    main()

