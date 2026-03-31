from algorithms.classification_breast_cancer import run_classification_breast_cancer
from algorithms.regression_californi import run_regression_california  # si tu l'as créé
from algorithms.lda_wine import run_lda
from algorithms.kmeans_iris import run_kmeans, plot_kmeans
from algorithms.decision_tree_iris import run_decision_tree
from algorithms.regression_housing import run_regression
from algorithms.Svm import run_svm
from algorithms.knn_iris import run_knn, plot_knn
from algorithms.bayes_tennis import run_bayes
import time

if __name__ == "__main__":
    start_total = time.time()

    # Lancement de l'analyse ML
    run_classification_breast_cancer()

    end_total = time.time()
    print("\n===== Temps d'exécution total du module =====")
    print(f"{end_total - start_total:.2f} secondes")



































"""from algorithms.lda_wine import run_lda
from algorithms.kmeans_iris import run_kmeans, plot_kmeans
from algorithms.decision_tree_iris import run_decision_tree
from algorithms.regression_housing import run_regression
from algorithms.Svm import run_svm
from algorithms.knn_iris import run_knn, plot_knn
from algorithms.bayes_tennis import run_bayes

import time


def main():
    start = time.time()


    print("\n===== LDA WINE CLASSIFICATION =====")
    run_lda()

    print("\n===== KMEANS CLUSTERING =====")
    X, kmeans_model, labels = run_kmeans()
    plot_kmeans(X, kmeans_model, labels)

    print("\n===== DECISION TREE IRIS =====")
    run_decision_tree()

    print("\n===== LINEAR REGRESSION HOUSING =====")
    run_regression()

    print("\n===== SVM CLASSIFICATION =====")
    run_svm()

    print("\n===== KNN CLASSIFIER =====")
    # Exécution KNN
    X, y, knn_model = run_knn()
    # Visualisation
    plot_knn(X, y)

    print("\n===== BAYES CLASSIFIER =====")
    run_bayes()

    end = time.time()

    print("Temps d'exécution :", end - start, "secondes")






if __name__ == "__main__":
    main()
"""