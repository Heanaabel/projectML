from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np

def run_lda():
    wine = load_wine()
    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    print("Accuracy LDA Wine:", accuracy_score(y_test, y_pred))

    # Prédiction d'un nouveau vin
    nouveau_vin = np.array([[14.2, 1.7, 2.4, 15.6, 127, 2.8, 3.0, 0.28, 2.2, 5.6, 1.0, 3.3, 1065]])
    nouveau_vin_scaled = scaler.transform(nouveau_vin)
    prediction = lda.predict(nouveau_vin_scaled)
    print("Prédiction pour nouveau vin :", wine.target_names[prediction[0]])