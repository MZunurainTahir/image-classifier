# CNN Image Classifier — CIFAR-10 Full Pipeline
# ===============================================

# %% [1] Imports
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.append(os.path.join(os.getcwd(), ".."))
from src.model import (build_custom_cnn, build_transfer_resnet, build_transfer_mobilenet,
                        get_callbacks, get_data_augmentation, CIFAR_CLASSES)
from src.visualize import plot_3d_confusion_matrix, plot_3d_training_surface, plot_model_comparison

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# %% [2] Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test, 10)
print(f"Train: {x_train.shape} | Test: {x_test.shape}")

# Visualize samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i]); ax.set_title(CIFAR_CLASSES[int(y_train[i])]); ax.axis("off")
plt.suptitle("CIFAR-10 Sample Images"); plt.tight_layout()
plt.savefig("../reports/figures/cifar10_samples.png", dpi=150); plt.show()

# %% [3] Data Augmentation
augmentation = get_data_augmentation()

# %% [4] Custom CNN
cnn = build_custom_cnn()
cnn.compile(optimizer=keras.optimizers.Adam(1e-3),
            loss="categorical_crossentropy", metrics=["accuracy"])
cnn.summary()

history_cnn = cnn.fit(
    augmentation(x_train), y_train_cat,
    epochs=50, batch_size=64,
    validation_data=(x_test, y_test_cat),
    callbacks=get_callbacks("custom_cnn"),
    verbose=1
)

# Training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.plot(history_cnn.history["loss"], label="train"); plt.plot(history_cnn.history["val_loss"], label="val"); plt.title("Loss"); plt.legend()
plt.subplot(1,2,2); plt.plot(history_cnn.history["accuracy"], label="train"); plt.plot(history_cnn.history["val_accuracy"], label="val"); plt.title("Accuracy"); plt.legend()
plt.savefig("../reports/figures/cnn_training.png", dpi=150); plt.show()

# %% [5] Evaluate Custom CNN
cnn_loss, cnn_acc = cnn.evaluate(x_test, y_test_cat, verbose=0)
print(f"Custom CNN — Test Accuracy: {cnn_acc:.4f}")

y_pred = np.argmax(cnn.predict(x_test), axis=1)
print(classification_report(y_test.flatten(), y_pred, target_names=CIFAR_CLASSES))

# %% [6] 3D Confusion Matrix
cm = confusion_matrix(y_test.flatten(), y_pred)
plot_3d_confusion_matrix(cm).show()

# %% [7] 3D Training Surface
plot_3d_training_surface(history_cnn.history).show()

# %% [8] Transfer Learning — MobileNetV2 (faster)
mobile = build_transfer_mobilenet()
mobile.compile(optimizer=keras.optimizers.Adam(1e-4),
               loss="categorical_crossentropy", metrics=["accuracy"])

history_mobile = mobile.fit(
    augmentation(x_train), y_train_cat,
    epochs=30, batch_size=64,
    validation_data=(x_test, y_test_cat),
    callbacks=get_callbacks("mobilenet"),
    verbose=1
)
mobile_loss, mobile_acc = mobile.evaluate(x_test, y_test_cat, verbose=0)
print(f"MobileNetV2 — Test Accuracy: {mobile_acc:.4f}")

# %% [9] Model Comparison
results = {
    "Custom CNN":   {"test_acc": cnn_acc},
    "MobileNetV2":  {"test_acc": mobile_acc},
}
plot_model_comparison(results).show()

# %% [10] Grad-CAM
try:
    from src.gradcam import make_gradcam_heatmap, save_gradcam
    img = x_test[0:1]
    last_conv = [l.name for l in cnn.layers if "conv2d" in l.name][-1]
    heatmap = make_gradcam_heatmap(img, cnn, last_conv)
    save_gradcam((img[0]*255).astype("uint8"), heatmap, "../reports/figures/gradcam_example.jpg")
    print("Grad-CAM saved!")
except Exception as e:
    print(f"Grad-CAM error: {e}")

print("Pipeline complete!")
