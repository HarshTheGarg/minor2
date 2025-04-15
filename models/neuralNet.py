#!../.venv/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as functional

import pandas as pd
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, in_features=42, h1=20, h2=10, out_features=3):
        super().__init__()

        # fc -> Fully connected
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.out(x)
        
        return x

def train_neural_net(pathToCSV, model_path="neural_net.pt"):

    df = pd.read_csv(pathToCSV, header=None).drop_duplicates()

    model = Model()

    X = df.iloc[:, 1:]
    y = df[0]

    X = X.values
    y = y.values

    X_train = torch.FloatTensor(X)
    y_train = torch.LongTensor(y)
    
    criterion = nn.CrossEntropyLoss()

    # lr -> Learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # print(model.parameters)

    epochs = 100
    losses = []
    
    for i in range(epochs):
        y_pred = model.forward(X_train)

        loss = criterion(y_pred, y_train)
        
        losses.append(loss.detach().numpy())

        if i % 10 == 0:
            print(f'Epoch {i} and loss: {loss}')

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    plt.plot(range(epochs), losses)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    # plt.show()
    plt.savefig("Epoch-vs-Losses.png")

    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval, y_test)
    
    print(loss)

    torch.save(model.state_dict(), model_path)
    # print(f"Model trained and saved as {model_path}")


def nn_predict(data, model_path="neural_net.pt"):

    model = Model()
    model.load_state_dict(torch.load(model_path))

    data = torch.tensor(data)

    with torch.no_grad():
        res = model(data)
    
    prediction = res.argmax().item()
    
    return prediction


if __name__ == '__main__':
    data = [0.0, 0.0, -0.236, 0.042, -0.484, 0.024, -0.672, 0.036, -0.818, 0.078,
            -0.563, -0.321, -0.769, -0.430, -0.896, -0.490, -1.0, -0.551, -0.478, -0.430,
            -0.678, -0.593, -0.806, -0.696, -0.909, -0.769, -0.357, -0.496, -0.503, -0.703,
            -0.606, -0.818, -0.696, -0.909, -0.2, -0.515, -0.236, -0.721, -0.278, -0.860,
            -0.321, -0.975]

    dataset_path = "./dataset.csv"


    train_neural_net(dataset_path)


    predicted_class = nn_predict(data)
    print(f"Predicted Class: {predicted_class}")