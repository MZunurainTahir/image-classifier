"""3D visualizations for CNN Image Classifier."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_3d_confusion_matrix(conf_matrix, class_names=None):
    if class_names is None:
        class_names = CIFAR_CLASSES
    n = len(conf_matrix)
    x = list(range(n))
    y = list(range(n))
    fig = go.Figure(data=[go.Surface(
        z=conf_matrix.tolist(),
        x=class_names, y=class_names,
        colorscale="Viridis",
        colorbar=dict(title="Count")
    )])
    fig.update_layout(
        title="3D Confusion Matrix — CIFAR-10",
        scene=dict(xaxis_title="Predicted", yaxis_title="Actual", zaxis_title="Count"),
        height=700
    )
    fig.write_html(FIG_DIR / "3d_confusion_matrix.html")
    print("Saved: 3d_confusion_matrix.html")
    return fig

def plot_3d_training_surface(history):
    epochs = list(range(1, len(history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=epochs, y=history["loss"], z=history["val_loss"],
                               mode="lines+markers", name="Train vs Val Loss",
                               line=dict(color="steelblue", width=4),
                               marker=dict(size=4)))
    fig.update_layout(
        title="3D Training Curve: Epoch x Train Loss x Val Loss",
        scene=dict(xaxis_title="Epoch", yaxis_title="Train Loss", zaxis_title="Val Loss"),
        height=600
    )
    fig.write_html(FIG_DIR / "3d_training_surface.html")
    return fig

def plot_model_comparison(results_dict):
    models = list(results_dict.keys())
    accuracies = [v["test_acc"] for v in results_dict.values()]
    fig = px.bar(x=models, y=accuracies, color=accuracies,
                 color_continuous_scale="Viridis",
                 title="Model Accuracy Comparison", text_auto=".2%",
                 labels={"x": "Model", "y": "Test Accuracy"})
    fig.write_html(FIG_DIR / "model_comparison.html")
    return fig
