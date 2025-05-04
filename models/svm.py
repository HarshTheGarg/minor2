#!../.venv/bin/python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np  
import pandas as pd

def train_svm(pathToCSV, model_path="svm_model.pkl"):
    df = pd.read_csv(pathToCSV, header=None).drop_duplicates()
    
    y = df[0]
    X = df.iloc[:, 1:]
    
    model = make_pipeline(
      StandardScaler(), 
      SVC(kernel='rbf', C=1.5, gamma='auto', probability=True, random_state=42)
    )

    model.fit(X, y)  
    
    joblib.dump(model, model_path)
    print(f"SVM Model trained and saved as {model_path}")

def svm_predict(data, model_path="svm_model.pkl"):
    model = joblib.load(model_path)
    data = np.array(data).reshape(1, -1)
    
    prediction = model.predict(data)[0]
    confidence = model.predict_proba(data).max()
    return prediction

def evaluate_model(pathToCSV, model_path="svm_model.pkl"):
    df = pd.read_csv(pathToCSV, header=None).drop_duplicates()
    
    y = df[0]
    X = df.iloc[:, 1:]
    
    model = joblib.load(model_path)
    accuracy = model.score(X, y)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':

    dataset_path = "./dataset.csv"
    train_svm(dataset_path)
    
    # test_data = [0.0, 0.0, -0.236, 0.042, -0.484, 0.024, -0.672, 0.036, -0.818, 0.078, -0.563, -0.321, -0.769, -0.430, -0.896, -0.490, -1.0, -0.551, -0.478, -0.430, -0.678, -0.593, -0.806, -0.696, -0.909, -0.769, -0.357, -0.496, -0.503, -0.703, -0.606, -0.818, -0.696, -0.909, -0.2, -0.515, -0.236, -0.721, -0.278, -0.860, -0.321, -0.975]
    
    # predicted_class, confidence = svm_predict(test_data)
    # print(f"Predicted Class: {predicted_class} with Confidence: {confidence * 100:.2f}%")
    
    # evaluate_model(dataset_path)
