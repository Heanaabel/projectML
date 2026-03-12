
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_kmeans():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_

    return X.values, kmeans, labels

def plot_kmeans(X, kmeans_model, labels=None):
    if labels is None:
        labels = kmeans_model.labels_

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 2], X[:, 3], c=labels, cmap='viridis', s=60, edgecolors='k', alpha=0.7)
    plt.scatter(kmeans_model.cluster_centers_[:, 2], kmeans_model.cluster_centers_[:, 3],
                c='red', marker='X', s=200, label='Centres')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title("KMeans Clustering Iris")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()