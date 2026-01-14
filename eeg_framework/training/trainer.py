import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .metrics import Evaluator

class Trainer:
    """
    Class to handle model training and evaluation.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device='cpu'):
        """
        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function.
            optimizer (optim.Optimizer): Optimizer.
            device (str): Device to train on ('cpu' or 'cuda').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.evaluator = Evaluator()

        self.model.to(self.device)

    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = self.evaluator.calculate_metrics(all_labels, all_preds)

        return epoch_loss, metrics

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = self.evaluator.calculate_metrics(all_labels, all_preds)

        return epoch_loss, metrics

    def train(self, num_epochs=10):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs.
        """
        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.evaluate()

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print("-" * 30)
