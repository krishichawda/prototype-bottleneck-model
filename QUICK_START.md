# Quick Start Guide - Prototype Bottleneck Model

## 🚀 Get Started in 5 Minutes

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from prototype_bottleneck import PrototypeBottleneckModel
import torch

# Create model
model = PrototypeBottleneckModel(
    input_dim=784,        # Input dimension (e.g., 28x28=784 for MNIST)
    bottleneck_dim=128,   # Bottleneck layer size
    num_prototypes=10,    # Number of prototypes to learn
    prototype_dim=784,    # Prototype dimension
    num_classes=10        # Number of output classes
)

# Train model
history = model.train_model(train_loader, epochs=50)

# Make interpretable predictions
predictions, prototype_weights = model.predict_with_explanations(x)

# Get detailed explanations
explanation = model.explain_prediction(x)
print(f"Prediction: {explanation['prediction']}")
print(f"Confidence: {explanation['confidence']:.3f}")
print(f"Top Prototypes: {explanation['top_prototypes'][:3]}")

# Visualize prototypes
model.visualize_prototypes()
```

### 3. Run Examples

```bash
# Test the model
python test_model.py

# Run MNIST example
python mnist_example.py

# Run interpretability demo
python interpretability_demo.py

# Explore visualization tools
python visualization_tools.py
```

## 🎯 Key Features

- **Interpretable Predictions**: Each decision is explained by prototype weights
- **Concept Learning**: Prototypes represent meaningful concepts in your data
- **Transparency**: Clear visibility into the decision-making process
- **Minimal Data**: Works efficiently with small datasets
- **Visualization**: Rich tools for understanding model behavior

## 📊 Model Architecture

```
Input → Encoder → Bottleneck → Prototypes → Decoder → Output
                ↓
            Prototype Layer
                ↓
            Attention Weights
```

## 🔍 Understanding the Output

- **Prototype Weights**: Show which prototypes influenced the decision
- **Confidence**: Model's confidence in the prediction
- **Reconstruction Error**: How well the model can reconstruct the input
- **Top Prototypes**: Most important prototypes for the decision

## 🛠️ Customization

### Adjust Model Parameters

```python
model = PrototypeBottleneckModel(
    input_dim=your_input_dim,
    bottleneck_dim=64,      # Smaller for more compression
    num_prototypes=20,      # More prototypes for complex data
    prototype_dim=your_prototype_dim,
    num_classes=your_num_classes,
    learning_rate=0.001     # Adjust learning rate
)
```

### Custom Training

```python
# Train with custom loss weights
history = model.train_model(
    train_loader, 
    epochs=100,
    reconstruction_weight=1.0,    # Weight for reconstruction loss
    classification_weight=2.0,    # Weight for classification loss
    verbose=True
)
```

## 📈 Monitoring Training

```python
# Plot training progress
model.plot_training_history()

# Check training history
print(f"Final Accuracy: {history['accuracy'][-1]:.2f}%")
print(f"Final Loss: {history['total_loss'][-1]:.4f}")
```

## 🔬 Advanced Usage

### Interpretability Analysis

```python
from interpretability_demo import InterpretabilityAnalyzer

analyzer = InterpretabilityAnalyzer(model, device)
analysis_results = analyzer.analyze_prototype_behavior(test_loader)
analyzer.visualize_prototype_specialization(analysis_results)
```

### Visualization Tools

```python
from visualization_tools import PrototypeVisualizer

visualizer = PrototypeVisualizer(model, device)
visualizer.create_prototype_analysis_dashboard(analysis_results)
visualizer.save_visualization_report(analysis_results)
```

## 🎯 Best Practices

1. **Start Small**: Begin with fewer prototypes and increase as needed
2. **Monitor Prototypes**: Ensure prototypes represent meaningful concepts
3. **Validate Interpretability**: Check that explanations make sense
4. **Use Visualization**: Leverage the visualization tools for insights
5. **Experiment**: Try different bottleneck dimensions and prototype counts

## 🚨 Troubleshooting

### Common Issues

1. **Low Accuracy**: Try increasing `num_prototypes` or `bottleneck_dim`
2. **Poor Interpretability**: Check if prototypes are diverse and meaningful
3. **Training Issues**: Adjust learning rate or loss weights
4. **Memory Issues**: Reduce batch size or model dimensions

### Getting Help

- Check the test script for working examples
- Review the MNIST example for a complete workflow
- Use the interpretability demo for advanced analysis

## 📚 Next Steps

1. **Run the MNIST Example**: See the model in action on real data
2. **Explore Interpretability**: Use the demo to understand model decisions
3. **Customize for Your Data**: Adapt the model to your specific use case
4. **Analyze Results**: Use visualization tools to gain insights

---

**Happy Modeling! 🎉**

The Prototype Bottleneck Model provides a powerful framework for building interpretable AI systems. Start with the examples and gradually customize for your specific needs.
