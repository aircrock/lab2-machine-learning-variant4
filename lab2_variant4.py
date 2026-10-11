
# -*- coding: utf-8 -*-
"""Лабораторная работа №2. PCA, SVD, ICA и кластеризация. Вариант 4."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kurtosis
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering,
    AgglomerativeClustering, DBSCAN, OPTICS, Birch, BisectingKMeans,
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, pairwise_distances,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
SEED = 42
PCA_FEATURES = ["a", "b", "c", "d", "e"]
CLUSTER_FEATURES = ["a", "b"]


def find_dataset(filename):
    for folder in (BASE_DIR, BASE_DIR.parent):
        path = folder / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"Файл {filename} не найден рядом со скриптом")


def load_data():
    # Вариант 4 в таблицах: PCA — столбцы S:X, кластеризация — J:L.
    pca_data = pd.read_excel(
        find_dataset("Lab2_Dataset_PCA.xlsx"),
        sheet_name=0, header=1, usecols="S:X",
    )
    clustering_data = pd.read_excel(
        find_dataset("Lab2_Dataset_Clustering.xlsx"),
        sheet_name=0, header=1, usecols="J:L",
    )

    # В общей таблице названия столбцов повторяются для каждого варианта.
    pca_data.columns = PCA_FEATURES + ["y"]
    clustering_data.columns = CLUSTER_FEATURES + ["y"]

    for frame in (pca_data, clustering_data):
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.dropna(inplace=True)

    return (
        pca_data[PCA_FEATURES].to_numpy(),
        pca_data["y"].to_numpy(dtype=int),
        clustering_data[CLUSTER_FEATURES].to_numpy(),
        clustering_data["y"].to_numpy(dtype=int),
    )


def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_correlation(X):
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    heatmap = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(5), PCA_FEATURES)
    ax.set_yticks(range(5), PCA_FEATURES)

    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center")

    fig.colorbar(heatmap, ax=ax)
    ax.set_title("Матрица корреляций")
    save_plot("01_correlation.png")
    return corr


def plot_scatter_matrix(X, y):
    fig, axes = plt.subplots(5, 5, figsize=(11, 11))
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            for cls in np.unique(y):
                values = X[y == cls]
                if i == j:
                    ax.hist(values[:, j], bins=16, alpha=0.5, density=True)
                else:
                    ax.scatter(values[:, j], values[:, i], s=9, alpha=0.6)
            if i == 4:
                ax.set_xlabel(PCA_FEATURES[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(PCA_FEATURES[i])
            else:
                ax.set_yticklabels([])
            ax.grid(alpha=0.15)

    fig.suptitle("Матрица разброса признаков", y=1.01)
    save_plot("02_scatter_matrix.png")


def plot_variance(values, method, filename):
    component_numbers = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(component_numbers, values * 100, label="Доля дисперсии")
    ax.plot(component_numbers, np.cumsum(values) * 100, "o-", label="Накопленная доля")
    ax.axhline(90, color="gray", linestyle="--", label="90%")
    ax.set_xticks(component_numbers)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Номер компоненты")
    ax.set_ylabel("Дисперсия, %")
    ax.set_title(f"Значимость компонент {method}")
    ax.legend()
    ax.grid(alpha=0.2)
    save_plot(filename)


def plot_ica_importance(importance, excess_kurtosis):
    order = np.argsort(importance)[::-1]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, importance[order] * 100, label="Вклад в реконструкцию")
    ax.set_xticks(x, [f"IC{i + 1}" for i in order])
    ax.set_xlabel("Компоненты ICA")
    ax.set_ylabel("Вклад, %")
    ax.grid(alpha=0.2, axis="y")

    ax2 = ax.twinx()
    ax2.plot(x, np.abs(excess_kurtosis[order]), "o-", color="darkred")
    ax2.set_ylabel("Модуль эксцесса")
    ax.set_title("Значимость и негауссовость компонент ICA")
    save_plot("05_ica_importance.png")


def plot_projection(scores, loadings, y, title, filename, x_label, y_label):
    fig, ax = plt.subplots(figsize=(8, 6))
    scale = np.ptp(scores[:, :2], axis=0)
    scale[scale == 0] = 1.0
    points = scores[:, :2] / scale

    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(points[mask, 0], points[mask, 1], s=25, alpha=0.75, label=f"y={cls}")

    for i, feature in enumerate(PCA_FEATURES):
        dx, dy = loadings[i, :2]
        ax.arrow(0, 0, dx, dy, color="firebrick", alpha=0.75,
                 head_width=0.035, head_length=0.04,
                 length_includes_head=True)
        ax.text(dx * 1.12, dy * 1.12, feature, color="darkred", fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    save_plot(filename)


def analyze_components(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    corr = plot_correlation(X)
    plot_scatter_matrix(X, y)

    pca = PCA().fit(X_scaled)
    pca_scores = pca.transform(X_scaled)
    plot_variance(pca.explained_variance_ratio_, "PCA", "03_pca_importance.png")
    plot_projection(
        pca_scores, pca.components_.T, y, "PCA: две главные компоненты",
        "06_pca_projection.png", "PC1", "PC2",
    )

    svd = TruncatedSVD(n_components=5, random_state=SEED).fit(X_scaled)
    svd_scores = svd.transform(X_scaled)
    plot_variance(svd.explained_variance_ratio_, "SVD", "04_svd_importance.png")
    plot_projection(
        svd_scores, svd.components_.T, y, "SVD: две главные компоненты",
        "07_svd_projection.png", "SVD1", "SVD2",
    )

    ica = FastICA(
        n_components=5, whiten="unit-variance", random_state=SEED,
        max_iter=5000, tol=1e-5,
    ).fit(X_scaled)
    ica_scores = ica.transform(X_scaled)
    mixing = ica.mixing_

    # ICA не сортирует компоненты по дисперсии. Для сравнения берём их
    # вклад в реконструкцию исходных стандартизованных признаков.
    energies = np.var(ica_scores, axis=0, ddof=1) * np.sum(mixing ** 2, axis=0)
    importance = energies / energies.sum()
    excess_kurtosis = kurtosis(ica_scores, axis=0, fisher=True, bias=False)
    order = np.argsort(importance)[::-1]
    plot_ica_importance(importance, excess_kurtosis)
    plot_projection(
        ica_scores[:, order[:2]], mixing[:, order[:2]], y,
        "ICA: две наиболее значимые компоненты", "08_ica_projection.png",
        f"IC{order[0] + 1}", f"IC{order[1] + 1}",
    )

    print("\n=== АНАЛИЗ КОМПОНЕНТ ===")
    print(f"Объектов: {len(X)}")
    print("\nМатрица корреляций:")
    print(pd.DataFrame(corr, index=PCA_FEATURES, columns=PCA_FEATURES).round(3).to_string())
    print("\nPCA:")
    for i, value in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {value:.2%}, накоплено {sum(pca.explained_variance_ratio_[:i + 1]):.2%}")
    print("\nSVD:")
    for i, value in enumerate(svd.explained_variance_ratio_):
        print(f"  SVD{i + 1}: {value:.2%}")
    print("\nICA (по убыванию вклада в реконструкцию):")
    for i in order:
        print(f"  IC{i + 1}: {importance[i]:.2%}, |эксцесс| = {abs(excess_kurtosis[i]):.3f}")


def count_clusters(labels):
    return len(set(labels.tolist()) - {-1})


def silhouette_without_noise(X, labels):
    mask = labels != -1
    n_clusters = len(np.unique(labels[mask]))
    if not 2 <= n_clusters < mask.sum():
        return np.nan
    return silhouette_score(X[mask], labels[mask])


def matching_accuracy(y_true, labels):
    # Номера кластеров не обязаны совпадать с эталонными номерами классов.
    observed = np.unique(labels)
    expected = np.unique(y_true)
    matrix = np.array([
        [np.sum((labels == a) & (y_true == b)) for b in expected]
        for a in observed
    ])
    rows, cols = linear_sum_assignment(-matrix)
    return matrix[rows, cols].sum() / len(y_true)


def create_models(X_scaled):
    distances = -pairwise_distances(X_scaled, metric="sqeuclidean")
    off_diagonal = distances[~np.eye(len(X_scaled), dtype=bool)]
    preference = float(np.quantile(off_diagonal, 0.001))

    return {
        "K-Means": KMeans(n_clusters=4, n_init=50, random_state=SEED),
        "Affinity Propagation": AffinityPropagation(
            damping=0.7, preference=preference, max_iter=1000,
            convergence_iter=30, random_state=SEED,
        ),
        "Mean Shift": MeanShift(bandwidth=0.9666666667, bin_seeding=True),
        "Spectral Clustering": SpectralClustering(
            n_clusters=4, affinity="nearest_neighbors", n_neighbors=10,
            assign_labels="kmeans", n_init=30, random_state=SEED,
        ),
        "Ward": AgglomerativeClustering(n_clusters=4, linkage="ward"),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=4, linkage="complete", metric="cosine",
        ),
        "DBSCAN": DBSCAN(eps=0.28, min_samples=4),
        "HDBSCAN": HDBSCAN(min_cluster_size=12, min_samples=2),
        "OPTICS": OPTICS(min_samples=5, xi=0.08, min_cluster_size=12),
        "Gaussian Mixture": GaussianMixture(
            n_components=4, covariance_type="full", n_init=50,
            init_params="k-means++", reg_covar=1e-5,
            random_state=SEED, max_iter=1000, tol=1e-5,
        ),
        "BIRCH": Birch(n_clusters=4, threshold=0.459, branching_factor=20),
        "Bisecting K-Means": BisectingKMeans(n_clusters=4, n_init=20, random_state=SEED),
    }


def draw_clusters(ax, X, labels, title):
    palette = plt.get_cmap("tab10")
    for label in sorted(np.unique(labels)):
        mask = labels == label
        if label == -1:
            ax.scatter(X[mask, 0], X[mask, 1], s=40, c="black", marker="x", label="Шум")
        else:
            ax.scatter(X[mask, 0], X[mask, 1], s=24,
                       color=palette(int(label) % 10), alpha=0.8,
                       label=f"Кластер {label}")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)


def plot_all_clusterings(X, results):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
    for ax, (name, labels) in zip(axes.flat, results.items()):
        noise = int(np.sum(labels == -1))
        subtitle = f"k={count_clusters(labels)}" + (f", шум={noise}" if noise else "")
        draw_clusters(ax, X, labels, f"{name}\n{subtitle}")
    for ax in axes[-1, :]:
        ax.set_xlabel("a")
    for ax in axes[:, 0]:
        ax.set_ylabel("b")
    fig.suptitle("Сравнение алгоритмов кластеризации (вариант 4)", y=1.01)
    save_plot("09_all_algorithms.png")


def plot_density_methods(X, results):
    names = ("DBSCAN", "HDBSCAN", "OPTICS")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        labels = results[name]
        draw_clusters(ax, X, labels, f"{name}: шум = {np.sum(labels == -1)}")
        ax.set_xlabel("a")
    axes[0].set_ylabel("b")
    save_plot("10_density_methods.png")


def plot_anomalies(X, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_clusters(ax, X, labels, "DBSCAN: кластеры и точки, отмеченные как шум")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.legend()
    save_plot("11_anomalies.png")


def plot_reference(X, y, labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    draw_clusters(axes[0], X, y, "Эталонные метки y")
    draw_clusters(axes[1], X, labels, "Gaussian Mixture")
    for ax in axes:
        ax.set_xlabel("a")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("b")
    save_plot("12_gmm_reference.png")


def analyze_clustering(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    results = {}
    metrics = []

    for name, model in create_models(X_scaled).items():
        labels = model.fit_predict(X_scaled)
        results[name] = labels
        metrics.append({
            "Алгоритм": name,
            "Кластеров": count_clusters(labels),
            "Шум": int(np.sum(labels == -1)),
            "Silhouette": silhouette_without_noise(X_scaled, labels),
            "ARI": adjusted_rand_score(y, labels),
            "NMI": normalized_mutual_info_score(y, labels),
            "Точность": matching_accuracy(y, labels),
        })

    plot_all_clusterings(X, results)
    plot_density_methods(X, results)
    plot_anomalies(X, results["DBSCAN"])
    plot_reference(X, y, results["Gaussian Mixture"])

    table = pd.DataFrame(metrics).sort_values("ARI", ascending=False)
    print("\n=== КЛАСТЕРИЗАЦИЯ ===")
    print(f"Объектов: {len(X)}")
    print("Silhouette рассчитывается только по точкам, не отмеченным как шум.\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    best = table.iloc[0]
    print(f"\nЛучший алгоритм по ARI: {best['Алгоритм']} (ARI = {best['ARI']:.3f})")
    print("Точность рассчитана после оптимального сопоставления номеров кластеров.")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Удаляем служебные таблицы, созданные предыдущей версией программы.
    for filename in (
        "clustering_metrics.csv", "component_importance.csv",
        "correlation_matrix.csv", "summary.txt",
    ):
        (OUTPUT_DIR / filename).unlink(missing_ok=True)

    X_pca, y_pca, X_clusters, y_clusters = load_data()
    analyze_components(X_pca, y_pca)
    analyze_clustering(X_clusters, y_clusters)
    print(f"\nГрафики сохранены в папке: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
