from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


def run_knn():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # division du dataset en test et train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Création du modèle KNN
    knn = KNeighborsClassifier(n_neighbors=3)

    # Entrainement du modèle
    knn.fit(X_train, y_train)

    # Prédiction
    y_pred = knn.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("KNN Accuracy:", acc)

    return X.values, y, knn


def plot_knn(X, y):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,2], X[:,3], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("KNN Dataset Visualization (Iris)")
    plt.grid(True)
    plt.show()