import cv2
import streamlit as st
from PIL import Image
from io import BytesIO
import base64
from retinaface import RetinaFace
import numpy as np

st.set_page_config(layout="wide", page_title="人の顔検出")

st.write("## Face Detect App")
st.write(
    ":cat: Try uploading an image to detect the humanfaces. Full quality images can be downloaded from the sidebar"
    )
    
st.write(
    "This app using RetinaFace algorithm. Special thanks to the [retina-face] [here](https://github.com/serengil/retinaface)"
)

print("test")

# Download the face dtect image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def detect_face(upload):
    image = Image.open(upload)
    
    try:
        cv2_img = cv2.imread(upload)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        col1.write("Original Image :camera:")
        col1.image(image)
    
    except:
        cv2_img=np.array(image)
        #cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        col1.write("Original Image :camera:")
        col1.image(image)
        

    #detect
    resp = RetinaFace.detect_faces(cv2_img, threshold = value)
    for key in resp:
        identity = resp[key]
        facial_area = identity["facial_area"]
        cv2.rectangle(cv2_img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (0, 255, 0), 2)
        
    col2.write("Detect face Image :wrench:")
    col2.image(cv2_img)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download result", convert_image(Image.fromarray(cv2_img)), "result.png", "image/png")


col1, col2 = st.columns(2)

st.sidebar.write("""
    ## Threshold
    """)

value= st.sidebar.slider(
        'Range',
        0.0, 1.0, value=0.7
    )

st.sidebar.write("## Upload and download :gear:")
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    detect_face(upload=my_upload)
else:
    detect_face("img3.jpg")
