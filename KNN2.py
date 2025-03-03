import pandas as pd
import numpy as np
import joblib

def train_knn_model(pathToCSV, model_path="knn_dataset.pkl"):
    df = pd.read_csv(pathToCSV, header=None).drop_duplicates()
    joblib.dump(df, model_path)
    print(f"KNN dataset saved as {model_path}")

def knn_predict(data, model_path="knn_dataset.pkl", k=5):
    df = joblib.load(model_path)
    y = df[0]
    X = df.iloc[:, 1:]

    dist, weights = [], []

    for i in range(len(X)):
        d = np.linalg.norm(X.iloc[i] - data)
        dist.append(d)
        if d == 0:
            weights.append(1000)
        else:
            weights.append(1 / (d ** 2))

    sortInd = np.argsort(dist)
    keys = y.unique()
    sums = {i: 0 for i in keys}

    for i in range(k):
        si = sortInd[i]
        sums[y.iloc[si]] += weights[si]

    maxy = max(sums.values())
    cla = [key for key in sums if sums[key] == maxy][0]
    return cla

if __name__ == '__main__':
    dataset_path = "./dataset.csv"


    train_knn_model(dataset_path)


    data = [0.0, 0.0, -0.236, 0.042, -0.484, 0.024, -0.672, 0.036, -0.818, 0.078,
            -0.563, -0.321, -0.769, -0.430, -0.896, -0.490, -1.0, -0.551, -0.478, -0.430,
            -0.678, -0.593, -0.806, -0.696, -0.909, -0.769, -0.357, -0.496, -0.503, -0.703,
            -0.606, -0.818, -0.696, -0.909, -0.2, -0.515, -0.236, -0.721, -0.278, -0.860,
            -0.321, -0.975]

    predicted_class = knn_predict(data)
    print(f"Predicted Class: {predicted_class}")
