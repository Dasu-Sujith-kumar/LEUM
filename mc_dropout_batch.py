import tensorflow as tf
import numpy as np
import cv2
import os
import csv

MODEL_PATH = "mobilenet_finetuned_combined.keras"
DATASET_DIR = "dataset-combined"
IMG_SIZE = (224,224)
MC_SAMPLES = 50
OUTPUT_CSV = "mc_dropout_results.csv"

model = tf.keras.models.load_model(MODEL_PATH)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def mc_dropout_predict(model, img, n_samples):
    preds = []

    for _ in range(n_samples):
        pred = model(img, training=True)
        preds.append(pred.numpy()[0][0])

    preds = np.array(preds)
    return preds.mean(), preds.std()


results = []

for label_name, class_label in [("original_all",0), ("tampered_all",1)]:

    folder = os.path.join(DATASET_DIR,label_name)

    for fname in os.listdir(folder):

        img_path = os.path.join(folder,fname)

        img = load_image(img_path)

        mean_pred, uncertainty = mc_dropout_predict(model,img,MC_SAMPLES)

        results.append([
            fname,
            label_name,
            class_label,
            round(mean_pred,4),
            round(uncertainty,4)
        ])


with open(OUTPUT_CSV,"w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "true_class",
        "true_label",
        "mean_tampered_probability",
        "uncertainty_std"
    ])

    writer.writerows(results)

print("MC Dropout results saved")