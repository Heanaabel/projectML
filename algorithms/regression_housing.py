import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def run_regression():
    data = pd.read_csv("housing.csv")
    X = data[['area','bedrooms','bathrooms']]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Linear Regression - aperçu des prédictions :")
    print(pd.DataFrame({"Prix réel": y_test.values[:5], "Prix prédit": y_pred[:5]}))