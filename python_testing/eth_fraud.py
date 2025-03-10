### data.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from python_testing.relu import ReLU, ConversionType

from python_testing.convolution import Convolution, QuantizedConv
# from python_testing.matrix_multiplication import QuantizedMatrixMultiplication
from python_testing.gemm import QuantizedGemm, Gemm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.utils.pytorch_helpers import ZKModel
import sys

import warnings
warnings.filterwarnings("ignore")

### eth_fraud.py
import torch.optim as optim

class EthereumFraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    """Loads and preprocesses the Ethereum fraud dataset, returning DataLoaders."""
    df = pd.read_csv("transaction_dataset.csv")

    # Drop non-numeric columns
    df = df.drop(columns=["Index", "Unnamed: 0", "Address"], errors='ignore')

    # Separate features and target
    y = df["FLAG"].astype(int).values
    X = df.drop(columns=["FLAG"], errors='ignore')

    # Handle categorical columns
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Fill missing values
    X.fillna(0, inplace=True)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Create DataLoaders
    batch_size = 32
    train_dataset = EthereumFraudDataset(X_train_tensor, y_train_tensor)
    val_dataset = EthereumFraudDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler

# class FraudDetectionCNN(nn.Module):
#     def __init__(self, input_dim):
#         super(FraudDetectionCNN, self).__init__()

#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

#         # self.fc1 = nn.Linear(32 * input_dim, 64)
#         self.fc1 = nn.Linear(1504, 64)
#         self.fc2 = nn.Linear(64, 1)

#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = x.unsqueeze(1).unsqueeze(1)  # Add channel dimension for Conv1d
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
class FraudDetectionCNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionCNN, self).__init__()

        # Use Conv2d instead of Conv1d
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Adjust the fully connected layer based on the new flattened size
        # Here we assume input_dim represents both height and width of the image or input data
        self.fc1 = nn.Linear(32 * input_dim, 64)  # Flattened size after Conv2d
        self.fc2 = nn.Linear(64, 1)  # Output layer for binary classification

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        # x is expected to be [batch_size, 1, height, width]
        x = self.relu(self.conv1(x))  # Shape: [batch_size, 16, height, width]
        x = self.relu(self.conv2(x))  # Shape: [batch_size, 32, height, width]
        
        x = self.flatten(x)  # Flatten to [batch_size, 32 * height * width]
        x = self.relu(self.fc1(x))  # Shape: [batch_size, 64]
        x = self.fc2(x)  # Shape: [batch_size, 1] (for binary classification)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y.float())
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


def evaluate_model(model, val_loader, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            logits = model(batch_X).squeeze()
            preds = (logits > threshold).int()
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(f"TP: {cm[1, 1]} | FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]} | TN: {cm[0, 0]}")
    print(f"\nBalanced Accuracy: {bal_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


def main():
    train_loader, val_loader, scaler = load_data()
    input_dim = next(iter(train_loader))[0].shape[1]

    model = FraudDetectionCNN(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    torch.save(model.state_dict(), "fraud_detection_model.pth")
    print("Model saved as 'fraud_detection_model.pth'")

    print("Validating on validation set:")
    evaluate_model(model, val_loader)


from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score

train_loader, val_loader, scaler = load_data()

    
class Eth(ZKModel):
    def __init__(self, file_name="model/fraud_detection_model.pth"):
        self.layers = {}
        self.name = "eth_fraud"

        self.scaling = 21

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FraudDetectionCNN(47).to(device)
        checkpoint = torch.load(file_name, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        self.model = model
        self.input_shape = [1, 47]

        self.data = load_data()
        for batch in self.data[1]:
            self.input_data = batch[0][0]
            self.labels = batch[1][0]
            break
        # torch.onnx.export(model, self.input_data.reshape(self.input_shape), f = "fraud_model.onnx")



    def get_model_params(self):
        exclude_keys = ['quantized', 'scaling']
        input_arr = torch.mul(2**self.scaling, self.input_data).reshape(self.input_shape).unsqueeze(1).unsqueeze(1).long()
        # input_arr = self.get_inputs(self.input_data_file).reshape(self.input_shape)
        inputs = {"input": input_arr.long().tolist()}
        weights = {}
        weights_2 = {}
        input = {}
        output = {}
        first_inputs = self.input_data.reshape(self.input_shape)
        outputs = self.read_output(self.model, first_inputs)
        
        layers = ["conv1", "relu", "conv2", "relu", "reshape", "fc1", "relu", "fc2"]
        # layers = ["conv1"]


        previous_output_tensor = input_arr

        for layer in layers:
            layer_params = {layer:{"quant":True}}
            if any(char.isdigit() for char in layer):
                l = self.model.__getattr__(layer)
                try:
                    layer_params = {layer:{"strides": l.stride}}
                except:
                    pass
                if layer == "fc2":
                    layer_params[layer]["quant"] = False


            else:
                l = layer
                if "reshape" in layer:
                    layer_params = {layer:{"shape": [-1,1504]}}

            # layer_params = self.model.__getattr__(layers[1])
            #Rework inputs to function
            if not layer in "reshape":
                (input, weight, output) = self.get_layer(input_arr, layer, l, **layer_params.get(layer, {"": None}))
                if weight:
                    weights.update({f"{layer}_" + key if key not in exclude_keys else key: value for key, value in weight.items()})
                input_arr = torch.LongTensor(input["input"])
                output_tensor = torch.LongTensor(output["output"])

                try:
                    self.check_4d_eq(input_arr,previous_output_tensor)
                except IndexError:
                    self.check_2d_eq(input_arr,previous_output_tensor)

                previous_output_tensor = output_tensor
                input_arr = output_tensor
            else:
                input_arr = torch.reshape(previous_output_tensor, layer_params["reshape"]["shape"])
                previous_output_tensor = input_arr
        weights["quantized"] = True


        
        for i in range(previous_output_tensor.shape[0]):
            for j in range(previous_output_tensor.shape[1]):
                error_margin = 0.1
                # x = previous_output_tensor[i][j]/(2**(2*self.scaling)) / outputs[i][j]
                # assert(x < (1 + error_margin))
                # assert(x > (1 - error_margin))
                print(outputs[i][j].item(), (previous_output_tensor[i][j]/(2**(2*self.scaling))).item())
                assert(abs(previous_output_tensor[i][j]/(2**(2*self.scaling)) - outputs[i][j]) < 0.0001)
        return inputs,[weights,weights_2],output

    

if __name__ == "__main__":
    # main()

    # Eth().run_circuit(demo = True)
    Eth()

