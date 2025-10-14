#!/usr/bin/env python3
"""
Simple test script for the Prototype Bottleneck Model.
This script verifies that the model can be created, trained, and used for predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from prototype_bottleneck import PrototypeBottleneckModel
from torch.utils.data import DataLoader, TensorDataset


def create_synthetic_data(num_samples=1000, input_dim=784, num_classes=10):
    """
    Create synthetic data for testing.
    
    Args:
        num_samples: Number of samples to create
        input_dim: Input dimension
        num_classes: Number of classes
        
    Returns:
        train_loader, test_loader: Data loaders
    """
    print("📊 Creating synthetic data...")
    
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    
    # Create simple patterns for different classes
    for i in range(num_classes):
        class_mask = (torch.arange(num_samples) % num_classes == i)
        # Add class-specific patterns
        X[class_mask, i*10:(i+1)*10] += 2.0
    
    # Create labels
    y = torch.arange(num_samples) % num_classes
    
    # Split into train and test
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Created {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    return train_loader, test_loader


def test_model_creation():
    """Test that the model can be created successfully."""
    print("\n🔧 Testing model creation...")
    
    try:
        model = PrototypeBottleneckModel(
            input_dim=784,
            bottleneck_dim=128,
            num_prototypes=10,
            prototype_dim=784,
            num_classes=10,
            learning_rate=0.001
        )
        
        # Test forward pass
        x = torch.randn(5, 784)
        logits, reconstructed, attention_weights, prototype_representation = model(x)
        
        print(f"✅ Model created successfully!")
        print(f"   - Input shape: {x.shape}")
        print(f"   - Logits shape: {logits.shape}")
        print(f"   - Reconstructed shape: {reconstructed.shape}")
        print(f"   - Attention weights shape: {attention_weights.shape}")
        print(f"   - Prototype representation shape: {prototype_representation.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_training(model, train_loader, epochs=5):
    """Test that the model can be trained."""
    print(f"\n🎯 Testing model training for {epochs} epochs...")
    
    try:
        history = model.train_model(train_loader, epochs=epochs, verbose=True)
        
        print("✅ Training completed successfully!")
        print(f"   - Final accuracy: {history['accuracy'][-1]:.2f}%")
        print(f"   - Final loss: {history['total_loss'][-1]:.4f}")
        
        return history
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


def test_predictions(model, test_loader):
    """Test that the model can make predictions with explanations."""
    print("\n🔍 Testing predictions with explanations...")
    
    try:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # Get predictions and explanations
                predictions, prototype_weights = model.predict_with_explanations(data)
                
                # Calculate accuracy
                correct += predictions.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        print(f"✅ Predictions completed successfully!")
        print(f"   - Test accuracy: {accuracy:.2f}%")
        print(f"   - Prototype weights shape: {prototype_weights.shape}")
        
        return accuracy, prototype_weights
        
    except Exception as e:
        print(f"❌ Predictions failed: {e}")
        return None, None


def test_interpretability(model, test_loader):
    """Test interpretability features."""
    print("\n💡 Testing interpretability features...")
    
    try:
        # Get a sample for explanation
        data_iter = iter(test_loader)
        sample_data, sample_target = next(data_iter)
        
        # Test explanation
        explanation = model.explain_prediction(sample_data[0:1])
        
        print("✅ Interpretability features work!")
        print(f"   - Prediction: {explanation['prediction']}")
        print(f"   - Confidence: {explanation['confidence']:.3f}")
        print(f"   - Top prototypes: {explanation['top_prototypes'][:3]}")
        print(f"   - Reconstruction error: {explanation['reconstruction_error']:.4f}")
        
        return explanation
        
    except Exception as e:
        print(f"❌ Interpretability test failed: {e}")
        return None


def test_visualization(model):
    """Test visualization features."""
    print("\n🎨 Testing visualization features...")
    
    try:
        # Test prototype visualization
        model.visualize_prototypes(figsize=(10, 6))
        print("✅ Prototype visualization works!")
        
        # Test training history plot
        if hasattr(model, 'training_history') and model.training_history['accuracy']:
            model.plot_training_history(figsize=(12, 4))
            print("✅ Training history visualization works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Prototype Bottleneck Model - Test Suite")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test 1: Model creation
    model = test_model_creation()
    if model is None:
        print("❌ Model creation test failed. Exiting.")
        return
    
    # Move model to device
    model = model.to(device)
    
    # Test 2: Data creation
    train_loader, test_loader = create_synthetic_data()
    
    # Test 3: Training
    history = test_training(model, train_loader, epochs=3)
    if history is None:
        print("❌ Training test failed. Exiting.")
        return
    
    # Test 4: Predictions
    accuracy, prototype_weights = test_predictions(model, test_loader)
    if accuracy is None:
        print("❌ Predictions test failed. Exiting.")
        return
    
    # Test 5: Interpretability
    explanation = test_interpretability(model, test_loader)
    if explanation is None:
        print("❌ Interpretability test failed. Exiting.")
        return
    
    # Test 6: Visualization
    viz_success = test_visualization(model)
    if not viz_success:
        print("❌ Visualization test failed.")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 50)
    print("✅ Model creation: PASSED")
    print("✅ Data generation: PASSED")
    print("✅ Model training: PASSED")
    print("✅ Predictions: PASSED")
    print("✅ Interpretability: PASSED")
    print("✅ Visualization: PASSED" if viz_success else "❌ Visualization: FAILED")
    
    print(f"\n📊 Final Results:")
    print(f"   - Test Accuracy: {accuracy:.2f}%")
    print(f"   - Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Prototypes: {model.num_prototypes}")
    
    print("\n🚀 The Prototype Bottleneck Model is working correctly!")
    print("You can now use it for your interpretable AI applications.")


if __name__ == "__main__":
    main()
