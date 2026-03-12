from algorithms.lda_wine import run_lda
from algorithms.kmeans_iris import run_kmeans
from algorithms.decision_tree_iris import run_decision_tree
from algorithms.regression_housing import run_regression
from algorithms.svm import run_svm   # <-- ici le changement

def main():
    print("\n===== LDA WINE CLASSIFICATION =====")
    run_lda()

    print("\n===== KMEANS IRIS CLUSTERING =====")
    run_kmeans()

    print("\n===== DECISION TREE IRIS =====")
    run_decision_tree()

    print("\n===== LINEAR REGRESSION HOUSING =====")
    run_regression()

    print("\n===== SVM CLASSIFICATION =====")
    run_svm()


if __name__ == "__main__":
    main()