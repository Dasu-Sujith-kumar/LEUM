# ============================================================
# RUN EVALUATION + PLOT GRAPHS (END-TO-END)
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image_dataset_from_directory

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "mobilenet_finetuned_patch.keras"
TEST_DIR = "dataset-patch"   # test folder only
RESULTS_DIR = "fine_results_patch"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

CLASS_NAMES = ["original_patch", "tampered_patch"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------
# FIX RANDOMNESS
# -------------------------------
tf.random.set_seed(SEED)
np.random.seed(SEED)

# -------------------------------
# LOAD MODEL
# -------------------------------
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# LOAD TEST DATA
# -------------------------------
print("📂 Loading test dataset...")

test_ds = image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Normalize
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# -------------------------------
# EVALUATE MODEL
# -------------------------------
print("🧪 Evaluating...")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)

print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# Save metrics
pd.DataFrame([{
    "test_accuracy": test_acc,
    "test_loss": test_loss
}]).to_csv(os.path.join(RESULTS_DIR, "final_test_metrics.csv"), index=False)

# -------------------------------
# PREDICTIONS
# -------------------------------
print("🔮 Predicting...")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    preds = (preds > 0.5).astype(int).flatten()
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# Raw CM
plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5
)
plt.title("Confusion Matrix", fontsize=14, weight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# Normalized CM
plt.figure(figsize=(7,6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5
)
plt.title("Normalized Confusion Matrix", fontsize=14, weight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png"), dpi=300)
plt.close()

# -------------------------------
# CLASSIFICATION REPORT
# -------------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(RESULTS_DIR, "classification_report.csv")
)

print("📊 Classification report saved")

# -------------------------------
# OPTIONAL: ACCURACY & LOSS GRAPH
# -------------------------------
if os.path.exists("training_history.npy"):
    history = np.load("training_history.npy", allow_pickle=True).item()

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(history["accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_loss.png"), dpi=300)
    plt.close()

    print("📈 Accuracy/Loss graph saved")

    # -------- EPOCH HEATMAPS --------
    # Accuracy Heatmap
    accuracy_data = np.array([history["accuracy"], history["val_accuracy"]])
    plt.figure(figsize=(12, 4))
    sns.heatmap(accuracy_data, annot=True, fmt=".3f", cmap="YlGn", 
                xticklabels=range(1, len(history["accuracy"])+1),
                yticklabels=["Train", "Val"], cbar_kws={"label": "Accuracy"})
    plt.title("Accuracy per Epoch (Heatmap)", fontsize=14, weight="bold")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_heatmap.png"), dpi=300)
    plt.close()

    # Loss Heatmap
    loss_data = np.array([history["loss"], history["val_loss"]])
    plt.figure(figsize=(12, 4))
    sns.heatmap(loss_data, annot=True, fmt=".3f", cmap="RdYlBu_r", 
                xticklabels=range(1, len(history["loss"])+1),
                yticklabels=["Train", "Val"], cbar_kws={"label": "Loss"})
    plt.title("Loss per Epoch (Heatmap)", fontsize=14, weight="bold")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_heatmap.png"), dpi=300)
    plt.close()

    print("📊 Epoch heatmaps saved")

print("\n🎯 DONE — ALL GRAPHS GENERATED")
print("📁 Saved in:", RESULTS_DIR)   