import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm


class PrototypeLayer(nn.Module):
    """
    Prototype layer that learns and stores prototypes.
    Each prototype represents a concept in the data.
    """
    def __init__(self, num_prototypes: int, prototype_dim: int, epsilon: float = 1e-4):
        super(PrototypeLayer, self).__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.epsilon = epsilon
        
        # Initialize prototypes as learnable parameters
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distances to prototypes and attention weights.
        
        Args:
            x: Input tensor of shape (batch_size, prototype_dim)
            
        Returns:
            distances: Distances to each prototype
            attention_weights: Soft attention weights over prototypes
        """
        # Compute squared Euclidean distances to prototypes
        # x: (batch_size, prototype_dim)
        # prototypes: (num_prototypes, prototype_dim)
        # distances: (batch_size, num_prototypes)
        distances = torch.cdist(x, self.prototypes, p=2) ** 2
        
        # Convert distances to similarity scores (smaller distance = higher similarity)
        similarities = torch.exp(-distances / (2 * self.epsilon))
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(similarities, dim=1)
        
        return distances, attention_weights


class BottleneckLayer(nn.Module):
    """
    Bottleneck layer that compresses representations and learns prototypes.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, num_prototypes: int, prototype_dim: int):
        super(BottleneckLayer, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        
        # Compression layers
        self.compress = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Prototype projection
        self.prototype_projection = nn.Linear(bottleneck_dim, prototype_dim)
        
        # Prototype layer
        self.prototype_layer = PrototypeLayer(num_prototypes, prototype_dim)
        
        # Reconstruction layers
        self.reconstruct = nn.Sequential(
            nn.Linear(prototype_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(bottleneck_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the bottleneck layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            compressed: Compressed representation
            prototype_representation: Prototype-based representation
            reconstructed: Reconstructed input
            attention_weights: Attention weights over prototypes
        """
        # Compress input
        compressed = self.compress(x)
        
        # Project to prototype space
        prototype_input = self.prototype_projection(compressed)
        
        # Get prototype distances and attention weights
        distances, attention_weights = self.prototype_layer(prototype_input)
        
        # Create prototype-based representation
        prototype_representation = torch.matmul(attention_weights, self.prototype_layer.prototypes)
        
        # Reconstruct input
        reconstructed = self.reconstruct(prototype_representation)
        
        return compressed, prototype_representation, reconstructed, attention_weights


class PrototypeBottleneckModel(nn.Module):
    """
    Complete prototype bottleneck model for interpretable AI.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, num_prototypes: int, 
                 prototype_dim: int, num_classes: int = 10, learning_rate: float = 0.001):
        super(PrototypeBottleneckModel, self).__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Bottleneck layer
        self.bottleneck = BottleneckLayer(input_dim, bottleneck_dim, num_prototypes, prototype_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(prototype_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(bottleneck_dim, num_classes)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'reconstruction_loss': [],
            'classification_loss': [],
            'total_loss': [],
            'accuracy': []
        }
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            logits: Classification logits
            reconstructed: Reconstructed input
            attention_weights: Attention weights over prototypes
            prototype_representation: Prototype-based representation
        """
        # Pass through bottleneck
        compressed, prototype_representation, reconstructed, attention_weights = self.bottleneck(x)
        
        # Classification
        logits = self.classifier(prototype_representation)
        
        return logits, reconstructed, attention_weights, prototype_representation
    
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, 
                    reconstruction_weight: float = 1.0, 
                    classification_weight: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """
        Compute the total loss.
        
        Args:
            x: Input data
            y: Target labels
            reconstruction_weight: Weight for reconstruction loss
            classification_weight: Weight for classification loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        logits, reconstructed, attention_weights, _ = self.forward(x)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, x)
        
        # Classification loss
        class_loss = self.classification_loss(logits, y)
        
        # Total loss
        total_loss = reconstruction_weight * recon_loss + classification_weight * class_loss
        
        loss_dict = {
            'reconstruction_loss': recon_loss.item(),
            'classification_loss': class_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, data_loader, reconstruction_weight: float = 1.0, 
                   classification_weight: float = 1.0) -> dict:
        """
        Train for one epoch.
        
        Args:
            data_loader: Data loader for training
            reconstruction_weight: Weight for reconstruction loss
            classification_weight: Weight for classification loss
            
        Returns:
            epoch_stats: Training statistics for the epoch
        """
        super().train()
        total_loss = 0
        total_recon_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            self.optimizer.zero_grad()
            
            # Flatten data if needed
            if len(data.shape) > 2:
                data = data.view(data.size(0), -1)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(data, target, reconstruction_weight, classification_weight)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['reconstruction_loss']
            total_class_loss += loss_dict['classification_loss']
            
            # Accuracy
            logits, _, _, _ = self.forward(data)
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        epoch_stats = {
            'reconstruction_loss': total_recon_loss / len(data_loader),
            'classification_loss': total_class_loss / len(data_loader),
            'total_loss': total_loss / len(data_loader),
            'accuracy': 100. * correct / total
        }
        
        return epoch_stats
    
    def train_model(self, data_loader, epochs: int = 100, reconstruction_weight: float = 1.0, 
             classification_weight: float = 1.0, verbose: bool = True) -> dict:
        """
        Train the model.
        
        Args:
            data_loader: Data loader for training
            epochs: Number of training epochs
            reconstruction_weight: Weight for reconstruction loss
            classification_weight: Weight for classification loss
            verbose: Whether to print progress
            
        Returns:
            training_history: Complete training history
        """
        if verbose:
            print(f"Training for {epochs} epochs...")
        
        for epoch in tqdm(range(epochs), disable=not verbose):
            epoch_stats = self.train_epoch(data_loader, reconstruction_weight, classification_weight)
            
            # Store history
            for key, value in epoch_stats.items():
                self.training_history[key].append(value)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={epoch_stats['total_loss']:.4f}, "
                      f"Accuracy={epoch_stats['accuracy']:.2f}%")
        
        return self.training_history
    
    def predict_with_explanations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with prototype explanations.
        
        Args:
            x: Input tensor
            
        Returns:
            predictions: Predicted classes
            prototype_weights: Attention weights showing prototype importance
        """
        self.eval()
        with torch.no_grad():
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            
            logits, _, attention_weights, _ = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
            
        return predictions, attention_weights
    
    def get_prototype_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the importance of each prototype for a given input.
        
        Args:
            x: Input tensor
            
        Returns:
            importance: Prototype importance scores
        """
        self.eval()
        with torch.no_grad():
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            
            _, _, attention_weights, _ = self.forward(x)
            
        return attention_weights
    
    def visualize_prototypes(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize the learned prototypes.
        
        Args:
            figsize: Figure size for the plot
        """
        prototypes = self.bottleneck.prototype_layer.prototypes.detach().cpu().numpy()
        
        # If prototypes are in a space that can be visualized (e.g., image space)
        if self.prototype_dim == self.input_dim:
            # Reshape for image visualization
            if self.input_dim == 784:  # MNIST
                prototypes_reshaped = prototypes.reshape(-1, 28, 28)
            else:
                prototypes_reshaped = prototypes
        else:
            # Use PCA to visualize high-dimensional prototypes
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            prototypes_2d = pca.fit_transform(prototypes)
            
            plt.figure(figsize=figsize)
            plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], s=100, alpha=0.7)
            for i, (x, y) in enumerate(prototypes_2d):
                plt.annotate(f'P{i}', (x, y), xytext=(5, 5), textcoords='offset points')
            plt.title('Prototype Visualization (PCA)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
            plt.show()
            return
        
        # Visualize prototypes as images
        n_prototypes = len(prototypes)
        cols = min(5, n_prototypes)
        rows = (n_prototypes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_prototypes):
            row = i // cols
            col = i % cols
            
            if self.input_dim == 784:  # MNIST
                axes[row, col].imshow(prototypes_reshaped[i], cmap='gray')
            else:
                axes[row, col].imshow(prototypes_reshaped[i])
            
            axes[row, col].set_title(f'Prototype {i}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_prototypes, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)):
        """
        Plot the training history.
        
        Args:
            figsize: Figure size for the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss plot
        axes[0].plot(self.training_history['total_loss'], label='Total Loss')
        axes[0].plot(self.training_history['reconstruction_loss'], label='Reconstruction Loss')
        axes[0].plot(self.training_history['classification_loss'], label='Classification Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.training_history['accuracy'])
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Prototype diversity (if available)
        if hasattr(self, 'prototype_diversity'):
            axes[2].plot(self.prototype_diversity)
            axes[2].set_title('Prototype Diversity')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Diversity Score')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Prototype diversity\nnot tracked', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Prototype Diversity')
        
        plt.tight_layout()
        plt.show()
    
    def explain_prediction(self, x: torch.Tensor, class_names: Optional[List[str]] = None) -> dict:
        """
        Provide detailed explanation for a prediction.
        
        Args:
            x: Input tensor
            class_names: Optional list of class names
            
        Returns:
            explanation: Dictionary with explanation details
        """
        self.eval()
        with torch.no_grad():
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            
            logits, reconstructed, attention_weights, prototype_representation = self.forward(x)
            prediction = torch.argmax(logits, dim=1)
            confidence = F.softmax(logits, dim=1).max(dim=1)[0]
            
            # Get top prototypes
            top_prototypes = torch.argsort(attention_weights, dim=1, descending=True)
            
        explanation = {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'prototype_weights': attention_weights.squeeze().cpu().numpy(),
            'top_prototypes': top_prototypes.squeeze().cpu().numpy(),
            'reconstruction_error': F.mse_loss(reconstructed, x).item()
        }
        
        if class_names:
            explanation['predicted_class'] = class_names[prediction.item()]
        
        return explanation
