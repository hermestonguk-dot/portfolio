"""
Feel free to replace this code with your own model training code. 
This is just a simple example to get you started.
"""

import numpy as np
import torch


def train(model, data_loader, criterion, optimizer, device):
    model.to(device)
    model.train()

    epoch_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    return avg_loss


def eval(model, data_loader, criterion, device):
    model.to(device)
    model.eval()

    targets = []
    predictions = []
    epoch_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            targets.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            epoch_loss += loss.item()

    accuracy = (np.array(predictions) == np.array(targets)).mean()
    avg_loss = epoch_loss / len(data_loader)
    return avg_loss, accuracy

