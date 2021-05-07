# Importing required libraries, obviously
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

# import dependencies
from IPython.display import display, Javascript, Image,clear_output
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
    
import av
import cv2
import numpy as np
import streamlit as st




def face_detect():
	print("Work on Progress")
  
    



    



from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)





WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)





# Loading pre-trained parameters for the cascade classifier
try:
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Face Detection
    classifier =load_model('Final_model.h5')  #Load model
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']  # Emotion that will be predicted
except Exception:
    st.write("Error loading cascade classifiers")



    
    


def about():
	st.write(
		'''
		**Haar Cascade** is an object detection algorithm.
		It can be used to detect objects in images or videos.
        
		The algorithm has four stages:
            
			1. Haar Feature Selection
            
			2. Creating  Integral Images
            
			3. Adaboost Training
            
			4. Cascading Classifiers
            
Read more :
    point_right: 
        https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')
def app_video_filters():
    """ Video transforms with OpenCV """

    class OpenCVVideoTransformer(VideoTransformerBase):
        type: Literal["noop", "cartoon", "edges", "rotate"]

        def __init__(self) -> None:
            self.type = "noop"

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
            elif self.type == "edges":
                # perform edge detection
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
            elif self.type == "rotate":
                # rotate image
                rows, cols, _ = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                img = cv2.warpAffine(img, M, (cols, rows))

            return img

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=OpenCVVideoTransformer,
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.type = st.radio(
            "Select transform type", ("noop", "cartoon", "edges", "rotate")
        )

    st.markdown(
        "This demo is based on "
        "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
        "Many thanks to the project."
    )


def main():
    
    activities = ["Introduction","Home", "Check Camera","About","Contact Us"]
    choice = st.sidebar.selectbox("Pick something Useful", activities)
    

    if choice == "Home":
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Emotion Recognition WebApp</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.title(":angry::dizzy_face::fearful::smile::pensive::open_mouth::neutral_face:")
        st.write("**Using the Haar cascade Classifiers**")
        st.write("Go to the About section from the sidebar to learn more about it.")
        st.write("**Instructions while using the APP**")
        st.write('''
                  1. Click on the Proceed button to start.
                 
                  2. WebCam window will open with a name of Emotion Detector automatically.If not found then, Have a look at the Taskbar.
                  
                  3. Make sure that camera shouldn't be used by any other app.
                  
                  4. Press "q" key to end the webcam window.
                  
                  5. Still webcam window didnot open,  go to Check Camera from the sidebar.''')
        
        if st.button("Proceed"):
            face_detect()
    
    
    elif choice == "Check Camera":
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Check Webcam is working or not</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write("**Instructions while Checking Camrea**")
        st.write('''
                  1. Click on  Start  to open webcam.
                 
                  2. If you have more than one camera , then select by using select device.
		  
		  3. Have some fun with your camera by choosing the options below.
                  
                  4. Click on  Stop  to end.
                  
                  5. Still webcam window didnot open,  Contact Us.''')
        app_video_filters()
        
        
        
        
        
        
        
    elif choice == "About":
        
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Haar Cascade Object Detection</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        about()
    elif choice=="Contact Us":
        with st.form(key='my_form'):
            text_input = st.text_input(label='Enter some text')
            submit_button = st.form_submit_button(label='Submit')
        st.write('''
                  Email:- soumya1999rta@gmail.com.
                 
                  Linkedin:-https://www.linkedin.com/in/soumya-ranjan-mishra-715674180/
                  
                  ''')
        
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:0.25px">
    <h2 style="color:white;text-align:center;">Copyright © 2021 | Soumya Ranjan Mishra </h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
    elif choice=="Introduction":
         html_temp = """
    <body style="background-color:red;">
    <div style="background-image: url('https://images.unsplash.com/photo-1542281286-9e0a16bb7366');padding:150px">
    <h2 style="color:red;text-align:center;">YOUR EMOTION REFLECTS YOUR PERSONALITY.</h2>
    <h2 style="color:white;text-align:center;">To Know your emotion proceed to Home from the side bar.</h2>
    </div>
    </body>
        """
         st.markdown(html_temp, unsafe_allow_html=True)
        
        
  

if __name__ == "__main__":
    main()


