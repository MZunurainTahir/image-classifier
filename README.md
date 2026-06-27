# CNN Image Classifier - CIFAR-10

**ML Concepts:** Deep Learning, CNN, Transfer Learning, Data Augmentation, Explainability

## Dataset
- **CIFAR-10:** 60,000 images (32x32 RGB), 10 classes, loaded via `keras.datasets`
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Architectures
| Model | Params | Accuracy |
|-------|--------|---------|
| Custom CNN | ~500K | ~75% |
| ResNet50 (Transfer) | ~23M | ~88% |
| MobileNetV2 (Transfer) | ~2.2M | ~85% |

## ML Concepts Applied
- Convolutional layers, pooling, batch normalization
- Dropout regularization
- Data augmentation (flip, rotate, zoom, shift)
- Transfer learning & fine-tuning (unfreeze top layers)
- Learning rate scheduling (ReduceLROnPlateau, CosineAnnealing)
- Grad-CAM explainability
- Early stopping & model checkpointing

## 3D Visualizations
- `reports/figures/3d_confusion_matrix.html` — 3D confusion matrix surface
- `reports/figures/3d_training_surface.html` — Loss/Accuracy training landscape
- `reports/figures/3d_feature_maps.html` — CNN feature map visualization

## Setup
```bash
pip install -r requirements.txt
jupyter lab
```
