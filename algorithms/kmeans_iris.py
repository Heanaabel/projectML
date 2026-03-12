from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def run_kmeans():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_

    plt.scatter(X.iloc[:,2], X.iloc[:,3], c=labels, cmap='viridis')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title("KMeans Clustering Iris")
    plt.show()