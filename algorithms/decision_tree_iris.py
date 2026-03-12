from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def run_decision_tree():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Decision Tree Report:")
    print(classification_report(y_test, y_pred))