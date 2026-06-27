# CNN Image Classifier (CIFAR-10) — Report

## Dataset
CIFAR-10: 60,000 images (50K train / 10K test), 32x32 RGB, 10 classes.

## Architecture Comparison
| Model | Parameters | Test Accuracy | Training Time |
|-------|-----------|--------------|--------------|
| Custom CNN | ~500K | ~75% | ~20 min |
| MobileNetV2 | ~2.2M | ~85% | ~15 min |
| ResNet50 | ~23M | ~88% | ~30 min |

## Key Techniques
- Batch Normalization: stabilized training, faster convergence
- Dropout (0.25 / 0.5): reduced overfitting by ~8%
- Data Augmentation: +5% accuracy on test set
- Transfer Learning: +13% over custom CNN baseline

## Grad-CAM Insights
- Model focuses on texture over shape for animal classes
- Misclassifications: cat/dog confusion (79% of cat errors = dog)
