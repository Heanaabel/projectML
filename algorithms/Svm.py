import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def run_svm():
    # Dataset : paniers clients
    dataset = [
        ['Lait', 'Oignon', 'Muscade', 'Haricots', 'Oeufs', 'Yaourt'],
        ['Aneth', 'Oignon', 'Muscade', 'Haricots', 'Oeufs', 'Yaourt'],
        ['Lait', 'Pomme', 'Haricots', 'Oeufs'],
        ['Lait', 'Unicorn', 'Maïs', 'Haricots', 'Yaourt'],
        ['Maïs', 'Oignon', 'Oignon', 'Haricots', 'Glace', 'Oeufs'],
        ['Lait', 'Oignon', 'Muscade', 'Haricots', 'Oeufs', 'Yaourt'],
        ['Lait', 'Pomme', 'Haricots', 'Oeufs'],
        ['Lait', 'Oignon', 'Muscade', 'Haricots', 'Oeufs', 'Yaourt'],
        ['Lait', 'Oignon', 'Muscade', 'Haricots', 'Oeufs', 'Yaourt'],
        ['Lait', 'Maïs', 'Haricots', 'Yaourt']
    ]

    # One-hot encoding
    all_items = sorted(set(item for panier in dataset for item in panier))
    df = pd.DataFrame(0, index=range(len(dataset)), columns=all_items)
    for i, panier in enumerate(dataset):
        for item in panier:
            df.loc[i, item] = 1

    # Création de la cible : Cuisinier = 1 si contient Oignon et Oeufs
    y = [1 if 'Oignon' in panier and 'Oeufs' in panier else 0 for panier in dataset]

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

    # Modèles
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    print("Prédictions sur l'ensemble test :")
    print("KNN :", knn.predict(X_test))
    print("SVM :", svm.predict(X_test))

    # Exemple nouveau client
    nouveau_panier = ['Lait', 'Haricots']
    nouveau_df = pd.DataFrame(0, index=[0], columns=df.columns)
    for produit in nouveau_panier:
        if produit in nouveau_df.columns:
            nouveau_df.loc[0, produit] = 1

    pred_knn = knn.predict(nouveau_df)[0]
    pred_svm = svm.predict(nouveau_df)[0]

    print(f"\nNouveau client {nouveau_panier} :")
    print(f"KNN -> {'Cuisinier' if pred_knn==1 else 'Non-cuisinier'}")
    print(f"SVM -> {'Cuisinier' if pred_svm==1 else 'Non-cuisinier'}")