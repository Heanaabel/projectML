import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_regression_california():
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split
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
    regressors = {
        "Random Forest": RandomForestRegressor(),
        "KNN": KNeighborsRegressor(),
        "SVM": SVR()
    }

    results_reg = {}

    for name, reg in regressors.items():
        model = Pipeline([
            ("preprocessing", preprocessor),
            ("regressor", reg)
        ])

        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results_reg[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Training_time": end - start
        }

    # Display
    print("\n===== CALIFORNIA HOUSING REGRESSION =====")
    for name, metrics in results_reg.items():
        print(f"\n{name} regression metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")