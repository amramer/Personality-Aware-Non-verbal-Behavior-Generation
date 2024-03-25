import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Setting up the layout and sidebar with profile photo and section titles
st.sidebar.title("Contents")
st.sidebar.markdown("""
- [Abstract](#abstract)
- [Demo Video](#demo-video)
- [Method](#method)
- [Applications](#applications)
""")

# Displaying the profile photo in the sidebar
image_url = "http://example.com/path_to_your_profile_image.png"  # replace with the actual image URL
response = requests.get(image_url)
# profile_img = Image.open(BytesIO(response.content))
# st.sidebar.image(profile_img, width=100, output_format='PNG', caption='Amr Farghaly')

# Main Content
st.title("Personality-Aware Non-verbal Behavior Generation in Dyadic Interactions")

# Authors
st.markdown("""
<div style="display: flex; justify-content: center; margin-bottom: 20px;">
    <div style="text-align: center;">
        <h4>Amr Farghaly</h4>
    </div>
</div>
""", unsafe_allow_html=True)

# Video
st.markdown("## Demo Video", unsafe_allow_html=True)
video_url = "http://example.com/path_to_your_video.mp4"  # replace with the actual video URL
st.video(video_url)

# Abstract
st.markdown("<a name='abstract'></a>", unsafe_allow_html=True)
st.subheader("Abstract")
st.markdown("""In dyadic interactions, non-verbal cues are vital for conveying emotions and
            intentions, significantly affecting communication perception and effectiveness. Previous
			methods for generating listener behavior responses have focused primarily on limited
			cues, like facial expressions, and overlooked the influence of personality traits on
			non-verbal behavior. This thesis introduces our approach to generating non-verbal
			behavior that captures a broader range of cues, including body language and hand
			gestures, and integrates the listener personality.""")

# Additional sections placeholder
st.markdown("<a name='method'></a>", unsafe_allow_html=True)
st.header("Method")
st.write("Methodology details go here.")

st.markdown("<a name='applications'></a>", unsafe_allow_html=True)
st.header("Applications")
st.write("Application details go here.")
