# Developing a streamlit aap having video, audio and image
# importing libraries
import streamlit as st
from PIL import Image

# Title
st.write("""
        # Add Image, Video and Audio in streamlit web app
        """)

# add image
st.write("""
        ## Image Section:
        """)
#image1 = Image.open('06_additive-manufacturing.jpg')
#st.image(image1, use_column_width=True)

# Adding slider to control image pixel size
width = st.slider('What is the width in pixels?', 0, 700, 350)
image1 = Image.open('06_additive-manufacturing.jpg')
st.image(image1, caption='test', width=width)

# Add video
st.write("""
        ## Video Section:
        """)
video1 = open('06_additive_manufacturing.mp4', 'rb')
st.video(video1)

# add audio
#audio1 = open('06_audio.mp3', 'rb')
#st.audio(audio1)