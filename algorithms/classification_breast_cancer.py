import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

def run_classification_breast_cancer(save_best_model=True):
    data = load_breast_cancer()
    X, y = data.data, data.target

    #Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing pipeline
    numeric_features = list(range(X.shape[1]))
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features)
    ])

    # Models
    classifiers = {
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}
    best_model_name = None
    best_f1 = -1
    best_model_pipeline = None

    for name, clf in classifiers.items():
        model = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", clf)
        ])

        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        errors = np.sum(preds != y_test)

        results[name] = {
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Misclassified": errors,
            "Training_time": end - start
        }

        # Matrice de confusion
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {name}")
        plt.show()

        # Feature importance pour Random Forest
        if name == "Random Forest":
            importances = model.named_steps['classifier'].feature_importances_
            plt.figure(figsize=(10,5))
            plt.bar(range(len(importances)), importances)
            plt.title("Feature Importance - Random Forest")
            plt.show()

        # Save du meilleur modèle
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_pipeline = model

    # Affichage des métriques
    print("\n===== BREAST CANCER CLASSIFICATION =====")
    for name, metrics in results.items():
        print(f"\n{name} metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # Sauvegarde du meilleur modèle pour API
    if save_best_model and best_model_pipeline is not None:
        filename = f"best_model_{best_model_name.replace(' ','_')}.pkl"
        joblib.dump(best_model_pipeline, filename)
        print(f"\nBest model ({best_model_name}) saved as '{filename}'")