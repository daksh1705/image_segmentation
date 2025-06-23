import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import io

st.set_page_config(page_title="Image Segmentation with KMeans", layout="wide")
st.title("🎨 Image Segmentation using K-Means Clustering")

# Sidebar: Upload + K value
uploaded_file = st.sidebar.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
k = st.sidebar.slider("🎯 Number of Segments (K)", 2, 10, 4)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))

    # Flatten image for clustering
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_img = centers[labels.flatten()].reshape(img.shape)

    # Show output
    st.subheader("🖼️ Original vs Segmented")
    col1, col2 = st.columns(2)
    col1.image(img, caption="Original Image", use_column_width=True)
    col2.image(segmented_img, caption=f"Segmented Image (K={k})", use_column_width=True)

    # Download button
    st.download_button("⬇️ Download Segmented Image",
                       data=cv2.imencode('.png', cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))[1].tobytes(),
                       file_name="segmented_image.png",
                       mime="image/png")
else:
    st.info("👈 Upload an image to get started.")
