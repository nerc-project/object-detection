# Import the dependencies we need to run the code.
import os
import cv2
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image
from remote_infer_rest import ort_v5

# Parameters
# Get a few environment variables. These are so we:
# - Know what endpoint we should request
# - Set server name and port for Streamlit
MODEL_NAME = os.getenv("MODEL_NAME", "yolo")                # You need to manually set this with an environment variable
REST_URL = os.getenv("INFERENCE_ENDPOINT")                  # You need to manually set this with an environment variable
INFER_URL = f"{REST_URL}/v2/models/{MODEL_NAME}/infer"

CLASSES_FILE = "coco.yaml"
IMAGE_DIR = "./images"

# Ensure the "images" folder exists
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Inference parameters
CONFIDENCE_THRESHOLD = 0.4  # Detections with lower scores won't be retained
IOU_THRESHOLD = 0.6  # Intersection over Union threshold for overlapping boxes

# Streamlit UI for image upload
st.title("YOLO Object Detection Application")
st.write("Upload an image to perform object detection using YOLO.")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to the images directory
    uploaded_image_path = os.path.join(IMAGE_DIR, uploaded_file.name)
    with open(uploaded_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run inference
    infer = ort_v5(uploaded_image_path, INFER_URL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, 640, CLASSES_FILE)
    img, out, result = infer()

    # Check if the predictions tensor is empty
    if out.size(0) == 0:
        st.write("No objects detected in the image.")
    else:
        st.write(f"{result}")
        st.write("Predictions:")
        st.write(out)
        st.write(
            "Format: each detection is a float64 array shaped as [top_left_corner_x, "
            "top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y, confidence, class_index]"
        )
        st.write("The coordinates are relative to a letterboxed representation of the image of size 640x640")

        # Process image and display it
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig = plt.gcf()
        fig.set_size_inches(24, 12)
        plt.axis("off")
        plt.imshow(img)
        st.pyplot(fig)
