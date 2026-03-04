import tensorflow as tf

model = tf.keras.models.load_model("mobilenet_finetuned_combined.keras")
model.summary()