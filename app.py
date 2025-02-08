import streamlit as st
import cv2
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

def detect_license_plates_and_recognize_text(image_path):
    model = YOLO(r'D:\Desktop\Soul page\runs\detect\train3\weights\best.pt')

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        st.error("Failed to load image. Please ensure the file format is supported.")
        return None, None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_image = image_rgb.copy()

    # Run inference
    results = model(image_path, conf=0.5)

    cropped_plates = []
    recognized_texts = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plate = image_rgb[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            cropped_plates.append(plate)
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
            result = reader.readtext(plate_gray)
            recognized_text = ''.join([text[1] for text in result])
            recognized_texts.append(recognized_text)

    return result_image, cropped_plates, recognized_texts


def main():
    st.set_page_config(layout="wide")
    st.title("License Plate Detection & Recognition App 🚗")

    left_col, right_col = st.columns([0.3, 0.7])

    with left_col:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a car image (Max size: 20 MB)", type=["jpg", "jpeg", "png"], help="Upload an image of a car with a visible license plate.")
        analyze_button = st.button("Analyze Image")

    if uploaded_file and analyze_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        with right_col:
            st.subheader("Results")
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            result_image, cropped_plates, recognized_texts = detect_license_plates_and_recognize_text(temp_file_path)

            if result_image is not None:
                st.image(result_image, caption="Detected License Plate(s)", use_container_width=True)

            if cropped_plates:
                st.image(cropped_plates[0], caption="Cropped License Plate", use_container_width=True)
                detected_text = recognized_texts[0] if recognized_texts else "No Text Detected"
                st.write(f"**License Plate Text:** {detected_text}")
            else:
                st.write("No license plate detected.")

if __name__ == "__main__":
    main()
