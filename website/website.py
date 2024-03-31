#######Imports and header########
import streamlit as st
import requests
from PIL import Image
from bio import *
from io import BytesIO

st.set_page_config(page_title = "Master Thesis", layout = "centered", page_icon = "👨🏻‍💼💻")

#######Sidebar########
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")

# Get the image from the URL
response = requests.get(info["Photo"])
img = Image.open(BytesIO(response.content))

# Create the HTML string for the circular image
photo_html = f'<div class="circle-image">\
    <a href="{info["Photo"]}" target="_blank">\
        <img src="{info["Photo"]}" alt="Profile Picture">\
    </a>\
</div>'

# Display the image and information in the sidebar
st.sidebar.markdown(photo_html, unsafe_allow_html=True)

st.sidebar.markdown(info["Sidebar Name"], unsafe_allow_html = True)

st.sidebar.markdown(info["Sidebar About"], unsafe_allow_html = True)


# Sidebar contents
st.sidebar.title("Contents")
st.sidebar.markdown("""
- [Abstract](#abstract)
- [Demo Video](#demo-video)
- [Method](#method)
- [Applications](#applications)
""")

# Main content
st.title("Personality-Aware Non-verbal Behavior Generation in Dyadic Interactions")

# Video
st.markdown("## Demo Video", unsafe_allow_html=True)
video_url = "http://example.com/path_to_your_video.mp4"
st.video(video_url)

# Abstract
st.markdown("<a name='abstract'></a>", unsafe_allow_html=True)
st.subheader("Abstract")
st.markdown("""
In dyadic interactions, non-verbal cues are vital for conveying emotions and intentions, significantly affecting communication perception and effectiveness. Previous methods for generating listener behavior responses have focused primarily on limited cues, like facial expressions, and overlooked the influence of personality traits on non-verbal behavior.

This thesis introduces our approach to generating non-verbal behavior that captures a broader range of cues, including body language and hand gestures, and integrates the listener’s personality.

Our methodology builds upon existing models by utilizing a transformer-based architecture that integrates detailed motion capture data and personality traits to produce more nuanced and realistic listener responses. This expansion allows for a more comprehensive simulation of dyadic interactions, where the listener’s non-verbal cues are not only reactive but also reflective of inherent personality characteristics.

We evaluate our model on a large face-to-face dyadic dataset using a set of metrics and user study. Our model demonstrates a notable ability to generate distinguishable non-verbal behaviors influenced by Extraversion. Participants were able to clearly distinguish between the introverted and extraverted listener avatars, indicating the model’s effectiveness in capturing personality-driven non-verbal cues.

Furthermore, the incorporation of personality traits into the model significantly improved the engagement and interaction quality between the speaker and listener compared to a personality-agnostic baseline.
""")


# Method
st.markdown("<a name='method'></a>", unsafe_allow_html=True)
st.header("Method")
st.write("Methodology details go here.")

# Applications
st.markdown("<a name='applications'></a>", unsafe_allow_html=True)
st.header("Applications")
st.write("Application details go here.")
