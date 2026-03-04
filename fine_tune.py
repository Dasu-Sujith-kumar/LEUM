import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np

# ===============================
# CONFIG (GOOD VALUES)
# ===============================
DATASET_DIR = "dataset-patch"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_FROZEN = 5
EPOCHS_FINETUNE = 10
SEED = 42

# ===============================
# DATA LOADER
# ===============================
def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["original_patch", "tampered_patch"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["original_patch", "tampered_patch"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED
    )

    # ✅ NORMALIZE - CRITICAL!
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

    return train_ds, val_ds


# ===============================
# MODEL
# ===============================
def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False  # freeze first

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)  # ✅ MC Dropout layer
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


# ===============================
# TRAINING PIPELINE
# ===============================
if __name__ == "__main__":

    train_ds, val_ds = load_datasets()

    # -------- Phase 1: Frozen training --------
    model, base_model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN
    )

    # -------- Phase 2: Fine-tuning --------
    print("[INFO] Fine-tuning last 30 layers...")

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # ✅ correct LR
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE
    )

    # ✅ Save training history
    np.save("training_history.npy", history.history)

    model.save("mobilenet_finetuned_patch.keras")
    print("✅ Correct fine-tuned model saved")  