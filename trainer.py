# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, lr=1e-3, max_grad_norm=5.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Adjusting Loss and Optimizer based on the desired criteria
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Performance log lists per epoch
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        self.best_val_acc = 0.0

    # Implementation of the Circle of Epochs
    def fit(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Activating Training Mode
            train_loss, train_acc = self._run_epoch(self.train_loader, is_training=True)
            # Enabling Validation/Test Mode
            val_loss, val_acc = self._run_epoch(self.test_loader, is_training=False)
            
            # Recording the final performance of the Epoch
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                # torch.save(self.model.state_dict(), 'best_model.pt') # Option to save the top model

    # Processing children separately
    def _run_epoch(self, data_loader, is_training):
        self.model.train(is_training)
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Using tqdm to display progress bar
        progress_bar = tqdm(data_loader, desc="Training" if is_training else "Validation")
        
        for text, lengths, labels in progress_bar:
            # Transferring variables to the processing device (GPU/CPU)
            text = text.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)
            
            # Managing gradients during training
            if is_training:
                self.optimizer.zero_grad()
                
            # Passing the data forward process through the model
            outputs = self.model(text, lengths)
            loss = self.criterion(outputs, labels)
            
            if is_training:
                loss.backward()
                # Clipping Gradients to Prevent Exploding Gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
            # Calculation of loss statistics and accuracy
            total_loss += loss.item() * text.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += text.size(0)
            
        epoch_loss = total_loss / total_samples
        epoch_acc = (correct_predictions / total_samples) * 100
        return epoch_loss, epoch_acc