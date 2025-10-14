import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class PrototypeVisualizer:
    """
    Comprehensive visualization tools for the Prototype Bottleneck Model.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def visualize_prototype_evolution(self, prototypes_history: List[np.ndarray], 
                                    epochs: List[int], figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize how prototypes evolve during training.
        
        Args:
            prototypes_history: List of prototype arrays at different epochs
            epochs: List of epoch numbers
            figsize: Figure size
        """
        print("🔄 Visualizing prototype evolution...")
        
        # Select key epochs for visualization
        if len(epochs) > 8:
            step = len(epochs) // 8
            selected_epochs = epochs[::step]
            selected_prototypes = prototypes_history[::step]
        else:
            selected_epochs = epochs
            selected_prototypes = prototypes_history
        
        n_epochs = len(selected_epochs)
        n_prototypes = selected_prototypes[0].shape[0]
        
        fig, axes = plt.subplots(n_epochs, n_prototypes, figsize=figsize)
        if n_epochs == 1:
            axes = axes.reshape(1, -1)
        
        for epoch_idx, (epoch, prototypes) in enumerate(zip(selected_epochs, selected_prototypes)):
            for proto_idx in range(n_prototypes):
                if prototypes.shape[1] == 784:  # MNIST
                    prototype_img = prototypes[proto_idx].reshape(28, 28)
                    axes[epoch_idx, proto_idx].imshow(prototype_img, cmap='gray')
                else:
                    axes[epoch_idx, proto_idx].imshow(prototypes[proto_idx])
                
                if epoch_idx == 0:
                    axes[epoch_idx, proto_idx].set_title(f'P{proto_idx}')
                if proto_idx == 0:
                    axes[epoch_idx, proto_idx].set_ylabel(f'Epoch {epoch}')
                
                axes[epoch_idx, proto_idx].axis('off')
        
        plt.suptitle('Prototype Evolution During Training', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_prototype_similarity_matrix(self, prototypes: np.ndarray, 
                                         figsize: Tuple[int, int] = (10, 8)):
        """
        Create a similarity matrix showing relationships between prototypes.
        
        Args:
            prototypes: Prototype array
            figsize: Figure size
        """
        print("🔗 Creating prototype similarity matrix...")
        
        # Calculate pairwise similarities
        n_prototypes = len(prototypes)
        similarity_matrix = np.zeros((n_prototypes, n_prototypes))
        
        for i in range(n_prototypes):
            for j in range(n_prototypes):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Cosine similarity
                    dot_product = np.dot(prototypes[i], prototypes[j])
                    norm_i = np.linalg.norm(prototypes[i])
                    norm_j = np.linalg.norm(prototypes[j])
                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(similarity_matrix, 
                   annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=[f'P{i}' for i in range(n_prototypes)],
                   yticklabels=[f'P{i}' for i in range(n_prototypes)])
        plt.title('Prototype Similarity Matrix')
        plt.xlabel('Prototype')
        plt.ylabel('Prototype')
        plt.show()
        
        return similarity_matrix
    
    def visualize_prototype_decision_boundaries(self, data: np.ndarray, labels: np.ndarray,
                                              prototype_weights: np.ndarray, 
                                              figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize decision boundaries based on prototype usage.
        
        Args:
            data: Input data (2D for visualization)
            labels: True labels
            prototype_weights: Prototype attention weights
            figsize: Figure size
        """
        print("🎯 Visualizing prototype decision boundaries...")
        
        # Use PCA to reduce to 2D if needed
        if data.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
        else:
            data_2d = data
        
        # Create subplots for different prototypes
        n_prototypes = prototype_weights.shape[1]
        cols = min(4, n_prototypes)
        rows = (n_prototypes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for proto_idx in range(n_prototypes):
            row = proto_idx // cols
            col = proto_idx % cols
            
            # Scatter plot colored by prototype weight
            scatter = axes[row, col].scatter(data_2d[:, 0], data_2d[:, 1], 
                                           c=prototype_weights[:, proto_idx], 
                                           cmap='viridis', alpha=0.7)
            axes[row, col].set_title(f'Prototype {proto_idx} Influence')
            axes[row, col].set_xlabel('Component 1')
            axes[row, col].set_ylabel('Component 2')
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[row, col])
        
        # Hide empty subplots
        for i in range(n_prototypes, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Prototype Decision Boundaries', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_interactive_prototype_explorer(self, prototypes: np.ndarray, 
                                            prototype_weights: np.ndarray,
                                            data: Optional[np.ndarray] = None,
                                            labels: Optional[np.ndarray] = None):
        """
        Create an interactive prototype explorer using Plotly.
        
        Args:
            prototypes: Prototype array
            prototype_weights: Prototype attention weights
            data: Optional input data for visualization
            labels: Optional labels for data
        """
        print("🔍 Creating interactive prototype explorer...")
        
        n_prototypes = prototypes.shape[0]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prototype Visualization', 'Prototype Usage Distribution',
                          'Prototype Similarity Network', 'Sample-Prototype Relationships'),
            specs=[[{"type": "image"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Prototype visualization (if image-like)
        if prototypes.shape[1] == 784:  # MNIST
            # Show first few prototypes as images
            for i in range(min(4, n_prototypes)):
                prototype_img = prototypes[i].reshape(28, 28)
                fig.add_trace(
                    go.Image(z=prototype_img, name=f'P{i}'),
                    row=1, col=1
                )
        
        # 2. Prototype usage distribution
        avg_usage = prototype_weights.mean(axis=0)
        fig.add_trace(
            go.Bar(x=[f'P{i}' for i in range(n_prototypes)], 
                  y=avg_usage, name='Average Usage'),
            row=1, col=2
        )
        
        # 3. Prototype similarity network (simplified)
        if data is not None and labels is not None:
            # Use PCA for 2D visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            
            # Color by class
            unique_labels = np.unique(labels)
            colors = px.colors.qualitative.Set3[:len(unique_labels)]
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                fig.add_trace(
                    go.Scatter(x=data_2d[mask, 0], y=data_2d[mask, 1],
                              mode='markers', name=f'Class {label}',
                              marker=dict(color=colors[i], opacity=0.7)),
                    row=2, col=1
                )
        
        # 4. Sample-prototype relationship heatmap
        fig.add_trace(
            go.Heatmap(z=prototype_weights[:100],  # Show first 100 samples
                      x=[f'P{i}' for i in range(n_prototypes)],
                      y=[f'Sample {i}' for i in range(min(100, len(prototype_weights)))],
                      colorscale='Blues'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive Prototype Explorer")
        fig.show()
    
    def visualize_prediction_explanations(self, images: torch.Tensor, 
                                        predictions: torch.Tensor,
                                        prototype_weights: torch.Tensor,
                                        true_labels: Optional[torch.Tensor] = None,
                                        num_samples: int = 6,
                                        figsize: Tuple[int, int] = (20, 12)):
        """
        Create detailed visualizations of prediction explanations.
        
        Args:
            images: Input images
            predictions: Model predictions
            prototype_weights: Prototype attention weights
            true_labels: Optional true labels
            num_samples: Number of samples to visualize
            figsize: Figure size
        """
        print(f"💡 Visualizing prediction explanations for {num_samples} samples...")
        
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(num_samples, 4, figsize=figsize)
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original image
            axes[i, 0].imshow(images[i].squeeze().cpu(), cmap='gray')
            title = f'Input\nPred: {predictions[i]}'
            if true_labels is not None:
                title += f'\nTrue: {true_labels[i]}'
            axes[i, 0].set_title(title)
            axes[i, 0].axis('off')
            
            # Prototype weights
            weights = prototype_weights[i].cpu().numpy()
            axes[i, 1].bar(range(len(weights)), weights)
            axes[i, 1].set_title('Prototype Weights')
            axes[i, 1].set_xlabel('Prototype')
            axes[i, 1].set_ylabel('Weight')
            
            # Top prototypes
            top_indices = np.argsort(weights)[-3:][::-1]
            top_weights = weights[top_indices]
            axes[i, 2].bar(range(3), top_weights)
            axes[i, 2].set_title('Top 3 Prototypes')
            axes[i, 2].set_xlabel('Prototype')
            axes[i, 2].set_ylabel('Weight')
            axes[i, 2].set_xticks(range(3))
            axes[i, 2].set_xticklabels([f'P{p}' for p in top_indices])
            
            # Prototype influence visualization
            # Show how much each prototype contributes to the decision
            influence = weights / weights.sum() * 100
            axes[i, 3].pie(influence, labels=[f'P{j}' for j in range(len(weights))],
                          autopct='%1.1f%%', startangle=90)
            axes[i, 3].set_title('Prototype Influence (%)')
        
        plt.suptitle('Detailed Prediction Explanations', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_prototype_analysis_dashboard(self, analysis_results: Dict, 
                                          figsize: Tuple[int, int] = (20, 15)):
        """
        Create a comprehensive dashboard for prototype analysis.
        
        Args:
            analysis_results: Dictionary containing analysis results
            figsize: Figure size
        """
        print("📊 Creating comprehensive prototype analysis dashboard...")
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        # 1. Prototype-class correlation heatmap
        if 'prototype_class_corr' in analysis_results:
            sns.heatmap(analysis_results['prototype_class_corr'], 
                       annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=[f'P{i}' for i in range(analysis_results['prototype_class_corr'].shape[1])],
                       yticklabels=[f'Class {i}' for i in range(10)],
                       ax=axes[0, 0])
            axes[0, 0].set_title('Prototype-Class Correlation')
        
        # 2. Prototype specialization scores
        if 'prototype_specialization' in analysis_results:
            specialization = analysis_results['prototype_specialization']
            axes[0, 1].bar(range(len(specialization)), specialization)
            axes[0, 1].set_title('Prototype Specialization')
            axes[0, 1].set_xlabel('Prototype')
            axes[0, 1].set_ylabel('Specialization Score')
        
        # 3. Prototype usage distribution
        if 'prototype_weights' in analysis_results:
            usage = analysis_results['prototype_weights'].mean(axis=0)
            axes[0, 2].bar(range(len(usage)), usage)
            axes[0, 2].set_title('Average Prototype Usage')
            axes[0, 2].set_xlabel('Prototype')
            axes[0, 2].set_ylabel('Usage')
        
        # 4. Confidence vs prototype usage
        if 'confidences' in analysis_results and 'prototype_weights' in analysis_results:
            confidences = analysis_results['confidences']
            usage = analysis_results['prototype_weights'].mean(axis=0)
            axes[1, 0].scatter(usage, confidences, alpha=0.6)
            axes[1, 0].set_title('Confidence vs Prototype Usage')
            axes[1, 0].set_xlabel('Average Usage')
            axes[1, 0].set_ylabel('Confidence')
        
        # 5. Prediction accuracy by class
        if 'predictions' in analysis_results and 'labels' in analysis_results:
            predictions = analysis_results['predictions']
            labels = analysis_results['labels']
            
            class_accuracy = []
            for class_idx in range(10):
                mask = labels == class_idx
                if mask.sum() > 0:
                    accuracy = (predictions[mask] == labels[mask]).mean()
                    class_accuracy.append(accuracy)
                else:
                    class_accuracy.append(0)
            
            axes[1, 1].bar(range(10), class_accuracy)
            axes[1, 1].set_title('Accuracy by Class')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_xticks(range(10))
        
        # 6. Prototype diversity over time (if available)
        if 'prototype_diversity_history' in analysis_results:
            diversity_history = analysis_results['prototype_diversity_history']
            epochs = range(len(diversity_history))
            axes[1, 2].plot(epochs, diversity_history)
            axes[1, 2].set_title('Prototype Diversity Evolution')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Diversity Score')
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Error analysis by prototype usage
        if 'prototype_weights' in analysis_results and 'predictions' in analysis_results and 'labels' in analysis_results:
            errors = predictions != labels
            error_usage = analysis_results['prototype_weights'][errors].mean(axis=0)
            correct_usage = analysis_results['prototype_weights'][~errors].mean(axis=0)
            
            x = np.arange(len(error_usage))
            width = 0.35
            
            axes[2, 0].bar(x - width/2, correct_usage, width, label='Correct', alpha=0.8)
            axes[2, 0].bar(x + width/2, error_usage, width, label='Errors', alpha=0.8)
            axes[2, 0].set_title('Prototype Usage: Correct vs Errors')
            axes[2, 0].set_xlabel('Prototype')
            axes[2, 0].set_ylabel('Usage')
            axes[2, 0].legend()
            axes[2, 0].set_xticks(x)
            axes[2, 0].set_xticklabels([f'P{i}' for i in range(len(error_usage))])
        
        # 8. Prototype activation patterns
        if 'prototype_weights' in analysis_results:
            # Show activation patterns for different samples
            sample_indices = np.random.choice(len(analysis_results['prototype_weights']), 
                                            min(10, len(analysis_results['prototype_weights'])), 
                                            replace=False)
            
            activation_patterns = analysis_results['prototype_weights'][sample_indices]
            sns.heatmap(activation_patterns, ax=axes[2, 1], cmap='Blues',
                       xticklabels=[f'P{i}' for i in range(activation_patterns.shape[1])],
                       yticklabels=[f'Sample {i}' for i in sample_indices])
            axes[2, 1].set_title('Sample Prototype Activation Patterns')
        
        # 9. Summary statistics
        axes[2, 2].axis('off')
        summary_text = f"""
        Model Summary:
        
        Total Samples: {len(analysis_results.get('predictions', []))}
        Accuracy: {np.mean(analysis_results.get('predictions', []) == analysis_results.get('labels', [])):.3f}
        Prototypes: {analysis_results.get('prototype_weights', np.array([])).shape[1]}
        Avg Confidence: {np.mean(analysis_results.get('confidences', [])):.3f}
        """
        axes[2, 2].text(0.1, 0.5, summary_text, transform=axes[2, 2].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.suptitle('Comprehensive Prototype Analysis Dashboard', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def save_visualization_report(self, analysis_results: Dict, output_file: str = 'visualization_report.html'):
        """
        Save a comprehensive HTML report with all visualizations.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_file: Output HTML file path
        """
        print(f"📄 Saving visualization report to {output_file}")
        
        # Create HTML content with embedded plots
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prototype Bottleneck Model - Visualization Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 40px; }
                .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 8px; border-left: 4px solid #2196F3; }
                .insight { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .plot-container { text-align: center; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Prototype Bottleneck Model - Visualization Report</h1>
                    <p>Comprehensive analysis and visualization of model interpretability</p>
                </div>
        """
        
        # Add summary metrics
        if 'predictions' in analysis_results and 'labels' in analysis_results:
            accuracy = np.mean(analysis_results['predictions'] == analysis_results['labels'])
            html_content += f"""
                <div class="section">
                    <h2>📊 Model Performance Summary</h2>
                    <div class="metric">
                        <strong>Accuracy:</strong> {accuracy:.2%}
                    </div>
                    <div class="metric">
                        <strong>Total Samples:</strong> {len(analysis_results['predictions']):,}
                    </div>
                    <div class="metric">
                        <strong>Prototypes:</strong> {analysis_results.get('prototype_weights', np.array([])).shape[1]}
                    </div>
                    <div class="metric">
                        <strong>Avg Confidence:</strong> {np.mean(analysis_results.get('confidences', [])):.3f}
                    </div>
                </div>
            """
        
        # Add prototype analysis
        if 'prototype_specialization' in analysis_results:
            html_content += """
                <div class="section">
                    <h2>🎯 Prototype Analysis</h2>
                    <div class="insight">
                        <strong>Key Insight:</strong> Prototypes show varying degrees of specialization for different classes.
                        This indicates the model learns meaningful, interpretable concepts.
                    </div>
                    <h3>Most Specialized Prototypes:</h3>
                    <table>
                        <tr><th>Rank</th><th>Prototype</th><th>Specialization Score</th><th>Primary Class</th></tr>
            """
            
            specialization = analysis_results['prototype_specialization']
            most_specialized = np.argsort(specialization)[::-1]
            
            for i, proto_idx in enumerate(most_specialized[:5]):  # Top 5
                if 'prototype_class_corr' in analysis_results:
                    primary_class = np.argmax(analysis_results['prototype_class_corr'][:, proto_idx])
                else:
                    primary_class = "N/A"
                
                html_content += f"""
                        <tr><td>{i+1}</td><td>P{proto_idx}</td><td>{specialization[proto_idx]:.3f}</td><td>Class {primary_class}</td></tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Add recommendations
        html_content += """
                <div class="section">
                    <h2>📈 Recommendations</h2>
                    <ul>
                        <li><strong>Monitor Prototype Diversity:</strong> Ensure prototypes cover diverse concepts in your data</li>
                        <li><strong>Use Explanations:</strong> Leverage prototype weights for model debugging and user trust</li>
                        <li><strong>Validate Interpretability:</strong> Verify that prototypes represent meaningful concepts</li>
                        <li><strong>Optimize Architecture:</strong> Adjust number of prototypes based on data complexity</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🔍 Next Steps</h2>
                    <ul>
                        <li>Run the interactive prototype explorer for detailed analysis</li>
                        <li>Use individual prediction explanations for error analysis</li>
                        <li>Monitor prototype evolution during training</li>
                        <li>Validate prototype interpretability with domain experts</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Visualization report saved to {output_file}")


def main():
    """
    Demo function showing how to use the visualization tools.
    """
    print("🎨 Prototype Bottleneck Model - Visualization Tools Demo")
    print("=" * 60)
    
    # This is a demo - in practice, you would load a trained model
    print("📝 Note: This is a demonstration of visualization tools.")
    print("To use with a real model, train the prototype bottleneck model first.")
    
    # Example usage structure
    print("\n🔧 Example usage:")
    print("""
    # Initialize visualizer
    visualizer = PrototypeVisualizer(model, device)
    
    # Analyze prototype behavior
    analysis_results = analyzer.analyze_prototype_behavior(test_loader)
    
    # Create visualizations
    visualizer.visualize_prototype_specialization(analysis_results)
    visualizer.create_interactive_prototype_explorer(prototypes, prototype_weights)
    visualizer.create_prototype_analysis_dashboard(analysis_results)
    visualizer.save_visualization_report(analysis_results)
    """)
    
    print("\n✅ Visualization tools are ready to use!")
    print("\n🎯 Key Features:")
    print("- Prototype evolution visualization")
    print("- Interactive prototype explorer")
    print("- Prediction explanation visualizations")
    print("- Comprehensive analysis dashboard")
    print("- HTML report generation")


if __name__ == "__main__":
    main()
