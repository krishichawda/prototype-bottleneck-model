import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prototype_bottleneck import PrototypeBottleneckModel
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def load_mnist_data(batch_size=64, train_split=0.8):
    """
    Load and prepare MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training
        
    Returns:
        train_loader, val_loader, test_loader: Data loaders
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load full dataset
    full_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Load test dataset
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        accuracy: Model accuracy
        predictions: Model predictions
        true_labels: True labels
        prototype_weights: Prototype attention weights
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []
    all_prototype_weights = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Flatten data
            data = data.view(data.size(0), -1)
            
            # Get predictions and explanations
            predictions, prototype_weights = model.predict_with_explanations(data)
            
            # Calculate accuracy
            correct += predictions.eq(target).sum().item()
            total += target.size(0)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())
            all_prototype_weights.append(prototype_weights.cpu().numpy())
    
    accuracy = 100. * correct / total
    all_prototype_weights = np.concatenate(all_prototype_weights, axis=0)
    
    return accuracy, all_predictions, all_true_labels, all_prototype_weights


def visualize_predictions_with_prototypes(model, data_loader, num_samples=5, device='cpu'):
    """
    Visualize predictions with prototype explanations.
    
    Args:
        model: The trained model
        data_loader: Data loader
        num_samples: Number of samples to visualize
        device: Device to run on
    """
    model.eval()
    
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Flatten images
    images_flat = images.view(images.size(0), -1)
    
    # Get predictions and explanations
    predictions, prototype_weights = model.predict_with_explanations(images_flat)
    
    # Visualize first few samples
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i].squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original (True: {labels[i]}, Pred: {predictions[i]})')
        axes[i, 0].axis('off')
        
        # Prototype weights
        weights = prototype_weights[i].cpu().numpy()
        axes[i, 1].bar(range(len(weights)), weights)
        axes[i, 1].set_title('Prototype Weights')
        axes[i, 1].set_xlabel('Prototype Index')
        axes[i, 1].set_ylabel('Weight')
        
        # Top prototypes
        top_prototypes = np.argsort(weights)[-3:][::-1]  # Top 3
        axes[i, 2].bar(range(3), weights[top_prototypes])
        axes[i, 2].set_title('Top 3 Prototypes')
        axes[i, 2].set_xlabel('Prototype Index')
        axes[i, 2].set_ylabel('Weight')
        axes[i, 2].set_xticks(range(3))
        axes[i, 2].set_xticklabels([f'P{p}' for p in top_prototypes])
    
    plt.tight_layout()
    plt.show()


def create_interactive_prototype_analysis(model, test_loader, device='cpu'):
    """
    Create interactive visualization of prototype analysis.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        device: Device to run on
    """
    # Get predictions and prototype weights for test set
    accuracy, predictions, true_labels, prototype_weights = evaluate_model(model, test_loader, device)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions,
        'correct': np.array(true_labels) == np.array(predictions)
    })
    
    # Add prototype weights
    for i in range(prototype_weights.shape[1]):
        df[f'prototype_{i}'] = prototype_weights[:, i]
    
    # Prototype usage analysis
    prototype_usage = prototype_weights.mean(axis=0)
    
    # Create interactive plot
    fig = go.Figure()
    
    # Add prototype usage bars
    fig.add_trace(go.Bar(
        x=[f'Prototype {i}' for i in range(len(prototype_usage))],
        y=prototype_usage,
        name='Average Usage',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Prototype Usage Analysis',
        xaxis_title='Prototype',
        yaxis_title='Average Weight',
        showlegend=False
    )
    
    fig.show()
    
    # Prototype-class correlation
    class_prototype_corr = np.zeros((10, prototype_weights.shape[1]))
    
    for class_idx in range(10):
        class_mask = np.array(true_labels) == class_idx
        if class_mask.sum() > 0:
            class_prototype_corr[class_idx] = prototype_weights[class_mask].mean(axis=0)
    
    # Create heatmap
    fig = px.imshow(
        class_prototype_corr,
        x=[f'P{i}' for i in range(prototype_weights.shape[1])],
        y=[f'Class {i}' for i in range(10)],
        title='Prototype-Class Correlation Heatmap',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title='Prototype',
        yaxis_title='Class'
    )
    
    fig.show()
    
    return df, prototype_usage, class_prototype_corr


def main():
    """
    Main function to run the MNIST example.
    """
    print("🚀 Prototype Bottleneck Model - MNIST Example")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n📊 Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("\n🤖 Initializing Prototype Bottleneck Model...")
    model = PrototypeBottleneckModel(
        input_dim=784,  # 28x28 MNIST images
        bottleneck_dim=128,
        num_prototypes=10,  # One prototype per digit class
        prototype_dim=784,  # Same as input for interpretable prototypes
        num_classes=10,
        learning_rate=0.001
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n🎯 Training model...")
    history = model.train_model(
        train_loader, 
        epochs=50, 
        reconstruction_weight=1.0, 
        classification_weight=1.0,
        verbose=True
    )
    
    # Plot training history
    print("\n📈 Plotting training history...")
    model.plot_training_history()
    
    # Evaluate on validation set
    print("\n🔍 Evaluating on validation set...")
    val_accuracy, val_predictions, val_labels, val_prototype_weights = evaluate_model(
        model, val_loader, device
    )
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_accuracy, test_predictions, test_labels, test_prototype_weights = evaluate_model(
        model, test_loader, device
    )
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(test_labels, test_predictions))
    
    # Confusion matrix
    print("\n🎯 Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Visualize prototypes
    print("\n🖼️ Visualizing learned prototypes...")
    model.visualize_prototypes()
    
    # Visualize predictions with prototype explanations
    print("\n🔍 Visualizing predictions with prototype explanations...")
    visualize_predictions_with_prototypes(model, test_loader, num_samples=5, device=device)
    
    # Interactive prototype analysis
    print("\n📊 Creating interactive prototype analysis...")
    df, prototype_usage, class_prototype_corr = create_interactive_prototype_analysis(
        model, test_loader, device
    )
    
    # Detailed explanation for a few examples
    print("\n💡 Detailed explanations for sample predictions...")
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    for i in range(3):
        explanation = model.explain_prediction(images[i:i+1].view(1, -1))
        print(f"\nSample {i+1}:")
        print(f"  True Label: {labels[i]}")
        print(f"  Predicted: {explanation['prediction']}")
        print(f"  Confidence: {explanation['confidence']:.3f}")
        print(f"  Top Prototypes: {explanation['top_prototypes'][:3]}")
        print(f"  Reconstruction Error: {explanation['reconstruction_error']:.4f}")
    
    print("\n✅ MNIST Example Complete!")
    print("\nKey Insights:")
    print("- The model learns interpretable prototypes representing different digit concepts")
    print("- Each prediction can be explained by showing which prototypes were most important")
    print("- The bottleneck layer forces the model to learn compressed, meaningful representations")
    print("- Prototype weights provide transparency into the decision-making process")


if __name__ == "__main__":
    main()
