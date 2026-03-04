import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np

DATASET_DIR = "dataset-combined"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["original_all", "tampered_all"],
        image_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["original_all", "tampered_all"],
        image_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED
    )

    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

    return train_ds, val_ds


def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    train_ds, val_ds = load_datasets()

    model = build_model()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    np.save(f"training_history_{DATASET_DIR}.npy", history.history)

    model.save("mobilenet_tampering_model_combined.keras")
    print("Model saved")