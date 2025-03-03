import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_random_forest(pathToCSV, model_path="random_forest_model.pkl"):

    df = pd.read_csv(pathToCSV, header=None).drop_duplicates()

    y = df[0]
    X = df.iloc[:, 1:]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)  

    joblib.dump(clf, model_path)
    print(f"Model trained and saved as {model_path}")


def rf_predict(data, model_path="random_forest_model.pkl"):

    clf = joblib.load(model_path)
    data = np.array(data).reshape(1, -1)

    prediction = clf.predict(data)[0]
    return prediction


if __name__ == '__main__':
    data = [0.0, 0.0, -0.236, 0.042, -0.484, 0.024, -0.672, 0.036, -0.818, 0.078,
            -0.563, -0.321, -0.769, -0.430, -0.896, -0.490, -1.0, -0.551, -0.478, -0.430,
            -0.678, -0.593, -0.806, -0.696, -0.909, -0.769, -0.357, -0.496, -0.503, -0.703,
            -0.606, -0.818, -0.696, -0.909, -0.2, -0.515, -0.236, -0.721, -0.278, -0.860,
            -0.321, -0.975]

    dataset_path = "./dataset.csv"


    train_random_forest(dataset_path)


    predicted_class = rf_predict(data)
    print(f"Predicted Class: {predicted_class}")
