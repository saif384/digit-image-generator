
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained generator model
model = load_model("mnist_digit_generator.h5")

st.title("🧠 Handwritten Digit Image Generator")
st.markdown("This app generates 5 diverse handwritten digit images using a GAN trained on the MNIST dataset.")

# Input: Choose digit (for labeling only; generation is not conditional)
digit = st.selectbox("Select a digit (0-9):", list(range(10)))

if st.button("Generate Images"):
    noise_dim = 100
    noise = tf.random.normal([5, noise_dim])
    generated_images = model.predict(noise)

    st.markdown(f"### Generated images of digit {digit}")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Sample {i+1}")
    st.pyplot(fig)
