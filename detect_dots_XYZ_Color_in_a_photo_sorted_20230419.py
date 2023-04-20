import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

max_contours = 10
# Create application title and file uploader widget.
st.title("Skymagic Photo to 2D model Tool")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

if img_file_buffer is not None:
    # Create a value input to get the max_dots from the inputs
    max_contours = st.number_input('Please input the drone number', 1, 30000)

    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    img_color = img.copy()
    img_out = img.copy()

    # Convert the photo to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to obtain a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area.
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filter the contours to obtain only the max_contours dots
    dots = []
    for idx in range(min(max_contours, len(contours_sorted))):
        area = cv2.contourArea(contours_sorted[idx])
        if area >= -0.1:  # and area < 200:
            x, y, w, h = cv2.boundingRect(contours_sorted[idx])
            center = (int(x + w / 2), int(y + h / 2))
            radius = int(max(w, h) / 2)
            dots.append(center)
            cv2.circle(img_out, center, radius, (0, 255, 0), 2)

    # Create placeholders to display input and output images.
    placeholders = st.columns(2)
    # Display Input image in the first placeholder.
    placeholders[0].image(img, channels='BGR')
    placeholders[0].text("Input Image (from creative team)")
    # Display dots image.
    placeholders[1].image(img_out, channels='BGR')
    placeholders[1].text("Output Image (green dots = drones)")

    # text
    text_contents = ''
    for dot in dots:
        color_x = (dot[0])
        color_y = (dot[1])
        text_contents = text_contents + f'{dot[0]},{dot[1]},{img_color[color_y, color_x, 2]},{img_color[color_y, color_x, 1]},{img_color[color_y, color_x, 0]}\n '

    # download_botton for the txt file
    img_name = img_file_buffer.name
    index_dot = img_name.find('.')
    name_wo = img_name[0:index_dot ]


    st.download_button('Download txt (XY position & RGB color)', text_contents, name_wo+'.txt')