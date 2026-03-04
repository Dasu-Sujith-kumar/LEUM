import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

DATASET_TYPE = "combined"

IMG_SIZE = (224, 224)

DATASET_DIR = f"dataset-{DATASET_TYPE}"

MODEL_PATH = f"mobilenet_finetuned_{DATASET_TYPE}.keras"

OUTPUT_DIR = f"gradcam_results_{DATASET_TYPE}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

LAST_CONV_LAYER = "out_relu"


def load_image(img_path):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMG_SIZE)

    img = img.astype("float32") / 255.0

    return img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


tampered_folder = os.path.join(DATASET_DIR, "tampered_all")

image_name = os.listdir(tampered_folder)[0]

img_path = os.path.join(tampered_folder, image_name)

img = load_image(img_path)

img_input = np.expand_dims(img, axis=0)

heatmap = make_gradcam_heatmap(img_input, model, LAST_CONV_LAYER)

heatmap = cv2.resize(heatmap, IMG_SIZE)

heatmap = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

superimposed_img = heatmap_color * 0.5 + (img * 255) * 0.5


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Tampered Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(superimposed_img.astype(np.uint8))
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.savefig(os.path.join(OUTPUT_DIR,"gradcam_result.png"))

plt.show()

print("GradCAM result saved")