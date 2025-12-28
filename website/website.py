#######Imports and header########
import streamlit as st
import requests
from PIL import Image
import pandas as pd
from bio import *
from io import BytesIO
import base64 

st.set_page_config(page_title = "Master Thesis", layout = "centered", page_icon = "👨🏻‍💼💻")

#######Sidebar########
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("website/style.css")

# Load the image from a local directory
try:
    with open(info["Photo"], "rb") as image_file:
        img_data = image_file.read()
        img_base64 = base64.b64encode(img_data).decode()

    # Create the HTML string for the circular image
    photo_html = f'<div class="circle-image">\
        <img src="data:image/jpeg;base64,{img_base64}" alt="Profile Picture">\
    </div>'
    
    # Display the image and information in the sidebar
    st.sidebar.markdown(photo_html, unsafe_allow_html=True)
except IOError:
    st.sidebar.error("Error: Unable to load image.")

st.sidebar.markdown(info["Sidebar Name"], unsafe_allow_html = True)

st.sidebar.markdown(info["Sidebar About"], unsafe_allow_html = True)

font_awesome = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" integrity="sha384-SZXxX4whJ79/gErwcOYf+zWLeJdY/qpuqC4cAa9rOGUstPomtqpuNWT9wdPEn2fk" crossorigin="anonymous" style="margin-top: -20px;">'
st.markdown(font_awesome, unsafe_allow_html=True)
st.sidebar.markdown(f'''<i class="fas fa-envelope" style="color: #0072B5; margin-right: 5px;"></i><a href="mailto:{info["Email"]}" style="color: #0072B5; text-decoration: none;">{info["Email"]}</a>''', unsafe_allow_html=True)



# Custom CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #1E1E1E; /* Set background color to black */
        color: #FFFFFF; /* Set text color to white */
        padding: 20px; /* Add some padding */
    }
    .sidebar .sidebar-content a {
        color: #0072B5; /* Set link color to blue */
        text-decoration: none; /* Remove underline from links */
    }
    .sidebar .sidebar-content a:hover {
        text-decoration: underline; /* Add underline on hover */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar contents
st.sidebar.title("Contents")
st.sidebar.markdown("""
    - [<span style="font-weight: bold;">Demo Video</span>](#demo-video)
    - [<span style="font-weight: bold;">Abstract</span>](#abstract)
    - [<span style="font-weight: bold;">Method</span>](#method)
    - [<span style="font-weight: bold;">Evaluation</span>](#evaluation)
    - [<span style="font-weight: bold;">Implementation</span>](#implementation)
    - [<span style="font-weight: bold;">Applications</span>](#applications)
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div id="master-thesis-note">
    <p style="font-size: 12px; font-style: italic;">
    Master Thesis Project:<br>
    This project was conducted as part of a Master's thesis research at DFKI and Max Planck Institute for Informatics in Saarbrücken.<br><br>
    Copyright Notice:<br>
    © Amr Amer, 2024
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown(
    """
    <h1 style='text-align: left; color: #f0f0f0; font-size: 44px; font-weight: bold; text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8); margin-top: -10%;margin-bottom: 5%;'>
        Personality-Aware Non-verbal Behavior Generation in Dyadic Interactions
    </h1>
    """,
    unsafe_allow_html=True
)

video_path = 'videos/website_intro.mp4'

# Read the video file
with open(video_path, 'rb') as video_file:
    video_bytes = video_file.read()

# Encode video bytes to base64
video_base64 = base64.b64encode(video_bytes).decode('utf-8')

# Construct the HTML for the video tag
# Path to the video file
video_path = 'website/videos/website_intro.mp4'

# Read the video file
with open(video_path, 'rb') as video_file:
    video_bytes = video_file.read()

# Encode video bytes to base64
video_base64 = base64.b64encode(video_bytes).decode('utf-8')

# Construct the HTML for the video tag
video_html = f'''
    <div class="video-container">
        <video width="100%" controls autoplay>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
'''

# Display the HTML content using st.markdown
st.markdown(video_html, unsafe_allow_html=True)

# Add a professional note beneath the video
st.markdown('''
<div class="note-container">
    <p>Welcome! Please watch the video above for an overview of my thesis website.</p>
</div>
''', unsafe_allow_html=True)

# Custom CSS to style the background and text
st.markdown('''
<style>
    body {
        background-color: black;
        color: white;
        font-family: Arial, sans-serif;
    }
    .video-container {
        margin: auto;
        max-width: 800px;
        border: 1px solid #444;
        border-radius: 5px;
        overflow: hidden;
    }
    .note-container {
        text-align: center;
        margin-top: 20px;
        font-size: 1.2em;
        background-color: rgba(51, 51, 51, 0.55); /* Adjust the alpha (opacity) value as needed */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
    }

    .note-container p {
        margin: 0;
    }
</style>
''', unsafe_allow_html=True)


# Video
st.markdown("## Demo Video", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Teaser GIF for the Results
st.image('website/Images/final-avatars.gif')

st.markdown(""" 
We have developed a multimodal generative model that accurately simulates dyadic conversations by generating listener avatars that respond to the speaker. Our model creates different listener behaviors based on specified personality traits, such as extroversion or introversion. For example, an extroverted agent may engage with the speaker through frequent smiling and animated body language, while an introverted agent responds more subtly.

**Demo: Generated Listener Avatars**

To showcase our model ability to generate personalized and adaptable responses, we present three listener avatars, each with distinct personality characteristics: Extrovert, Neutral, and Introvert. Additionally, we provide videos below demonstrating various speaker and listener avatars interacting to illustrate the model capabilities.

**Potential Applications**

Our technology holds significant potential for developing virtual therapist assistants tailored to individual preferences. This innovation can democratize mental health support by enhancing accessibility and engagement, allowing for personalized counseling that meets individual needs without the need for psychologist visits. Imagine having a virtual therapist at home, designed to accommodate different personality types and provide customized support.
""")




# Create two columns with a specified width
col1, col2 = st.columns([4, 4])

# Load the first video in the first column
with col1:
    video_file1 = open('videos/listener_39_vs_speaker_models_comp.mp4', 'rb')
    video_bytes1 = video_file1.read()

    # Use st.markdown with HTML and custom CSS class to control the width and height of the video container
    video_html = f'''
                   <div class="video-container">
                     <video controls>
                       <source src="data:video/mp4;base64,{base64.b64encode(video_bytes1).decode()}" type="video/mp4">
                       Your browser does not support the video tag.
                     </video>
                   </div>
                 '''
    st.markdown(video_html, unsafe_allow_html=True)

# Load the second video in the second column
with col2:
    video_file2 = open('videos/listener_141_vs_speaker_models_comp.mp4', 'rb')
    video_bytes2 = video_file2.read()

    # Use st.markdown with HTML and custom CSS class to control the width and height of the video container
    video_html = f'''
                   <div class="video-container">
                     <video controls>
                       <source src="data:video/mp4;base64,{base64.b64encode(video_bytes2).decode()}" type="video/mp4">
                       Your browser does not support the video tag.
                     </video>
                   </div>
                 '''
    st.markdown(video_html, unsafe_allow_html=True)

# Create two new columns with a specified width
col3, col4 = st.columns([4, 4])

# Load the third video in the first new column
with col3:
    video_file3 = open('website/videos/listener_182_vs_speaker_models_comp.mp4', 'rb')
    video_bytes3 = video_file3.read()

    # Use st.markdown with HTML and custom CSS class to control the width and height of the video container
    video_html = f'''
                   <div class="video-container">
                     <video controls>
                       <source src="data:video/mp4;base64,{base64.b64encode(video_bytes3).decode()}" type="video/mp4">
                       Your browser does not support the video tag.
                     </video>
                   </div>
                 '''
    st.markdown(video_html, unsafe_allow_html=True)

# Load the fourth video in the second new column
with col4:
    video_file4 = open('website/videos/listener_145_vs_speaker_models_comp.mp4', 'rb')
    video_bytes4 = video_file4.read()

    # Use st.markdown with HTML and custom CSS class to control the width and height of the video container
    video_html = f'''
                   <div class="video-container">
                     <video controls>
                       <source src="data:video/mp4;base64,{base64.b64encode(video_bytes4).decode()}" type="video/mp4">
                       Your browser does not support the video tag.
                     </video>
                   </div>
                 '''
    st.markdown(video_html, unsafe_allow_html=True)

# Abstract
st.markdown("<a name='abstract'></a>", unsafe_allow_html=True)
st.subheader("Abstract")
st.markdown("""
Non-verbal communication is a crucial aspect of dialogue and social interactions,  
as people use various cues such as facial expressions, eye gaze, and body movements to convey emotions. 
However, the way people express emotions can vary greatly depending on their personality traits. 

This topic is interesting due to its real-life applications, 
such as creating emotionally intelligent virtual agents for customer support and
virtual therapists in healthcare to provide more supportive and empathetic environment.

In this thesis, we present a new transformer-based architecture that integrates 
a broad range of cues, including body language and hand gestures, 
to generate non-verbal behavior in response to a speaker's talk, taking into 
account the listener's personality traits. Using the UDIVA dataset, 
our model outperforms existing base models that do not consider personality traits, 
achieving an FID score of 6.15 and P-FID score of 10.31 for facial expressions, 
and an FID score of 43.16 and P-FID score of 87.73 for body and hand gestures. 
In addition, participants in the experiment were able to correctly identify 
extroverted listener avatars in 86% of cases, confirming the model's ability to 
capture personality-driven non-verbal cues.
""")

st.markdown("""
---

<sub>UDIVA dataset description [here](https://chalearnlap.cvc.uab.es/dataset/41/description/).</sub>
""", unsafe_allow_html=True)
# Method
st.markdown("<a name='method'></a>", unsafe_allow_html=True)
st.header("Method")

# Data Preprocessing
st.subheader("1. Data Pre-Processing")
# Load the image
pre_processing = Image.open('Images/pre-processing-pipeline.png')

# Convert the image to base64
buffer = BytesIO()
pre_processing.save(buffer, format='PNG')
img_str = base64.b64encode(buffer.getvalue()).decode()

# Create the HTML string for the image container with increased height
image_html = f"""
<div class="image-container" style="text-align: center;">
    <img src="data:image/png;base64,{img_str}" alt="Pre-processing pipeline" style="height: 140px;"">
    <div style="font-size: 14px; margin-top: 8px; color: grey;">Dataset Preprocessing Pipeline</div>
    <div class="note-icon">🔍</div>
    <div class="note-text">Hover to enlarge</div>
    <div class="enlarged-image">
        <img src="data:image/png;base64,{img_str}" alt="Enlarged Model Architecture">
    </div>
</div>
"""

# Display the image container
st.markdown(image_html, unsafe_allow_html=True)

st.markdown("Our data preprocessing pipeline involves the following steps:")

# Define the pipeline steps
pipeline_steps = [
    "**Video Segmentation:** We start by segmenting dyadic interaction videos into separate speaker and listener streams. This is achieved using conversation transcript timestamps, resulting in individual speaker and listener videos.",
    "**Cropping Video Frames (ROI):** Next, we define a boundary box around each participant using dataset annotations that mark body, face, and hand joint landmarks. This box is defined by the extreme points: the shoulders (leftmost and rightmost), the head (topmost), and the torso (bottommost). This step allows us to accurately crop the video frames around the participants.",
    "**Feature Extraction (Audio & Motion):** We then extract the audio sequence from the speaker's video and convert it into a mel-spectrogram using Librosa, with a sampling rate of 16 kHz. For motion feature extraction, we use [PIXIE](https://github.com/yfeng95/PIXIE) to process both speaker and listener videos. This results in the extraction of SMPL-X parameters for each video frame, which include facial expression coefficients, 3D jaw and head rotation, and upper body parameters representing relative joint positions.",
    "**Postprocessing:** However, we noticed jittering in the motion sequences generated by PIXIE. To resolve this, we applied a dual filtering strategy on the extracted parameters for each video frame. First, we used a mean average filter within a window size of 7 frames to smooth the transition between frames. Then, to minimize abrupt motion jumps while retaining motion variability, we applied a median filter with a smaller window size of 3 frames.",
    "**Output:** The final output is a refined dataset containing speaker audio features and motion features for both speaker and listener, ready for model training."
]

# Display the first part of the pipeline steps up to the "Cropping Video Frames (ROI)" step
st.markdown('<div class="step-list">', unsafe_allow_html=True)
for step in pipeline_steps[:2]:
    st.markdown(f'- {step}')
st.markdown('</div>', unsafe_allow_html=True)

# Add CSS for the margin
st.markdown(
    """
    <style>
    .imagecol-container {
        margin-left: 150px; /* Adjust the value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the image container with the specified margin
st.markdown('<div class="imagecol-container">', unsafe_allow_html=True)

# Using columns to place images side by side
col1, col2 = st.columns(2)

with col1:
    st.image("website/Images/udivia_annotations_2.png", caption="Dataset Annotation Example 1", width=320)

with col2:
    st.image("website/Images/udivia_annotations_3.png", caption="Dataset Annotation Example 2", width=320)

# Closing the div container
st.markdown('</div>', unsafe_allow_html=True)

# Display the remaining pipeline steps
st.markdown('<div class="step-list">', unsafe_allow_html=True)
for step in pipeline_steps[2:]:
    st.markdown(f'- {step}')
st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
---

<sub>Learn more about PIXIE [here](https://github.com/yfeng95/PIXIE).</sub>
""", unsafe_allow_html=True)

# Model Pipeline
st.subheader("2. Model Architecture")

# Load the image
model_architecture = Image.open('Images/model-architecture.png')

# Convert the image to base64
buffer = BytesIO()
model_architecture.save(buffer, format='PNG')
img_str = base64.b64encode(buffer.getvalue()).decode()

# Create the HTML string for the image container
image_html = f"""
<div class="image-container">
    <img src="data:image/png;base64,{img_str}" alt="Model Architecture">
    <div style="font-size: 14px; margin-top: 8px; color: grey; text-align: center;">Model Architecture</div>
    <div class="note-icon">🔍</div>
    <div class="note-text">Hover to enlarge</div>
    <div class="enlarged-image">
        <img src="data:image/png;base64,{img_str}" alt="Enlarged Model Architecture">
    </div>
</div>
"""

# Add an introduction
st.markdown("We present a model that generates personalized non-verbal behaviors in interactive avatars based on the listener's personality traits. Using a transformer architecture and a vector quantized variational autoencoder (VQ-VAE), our model produces dynamic responses.")

# Display an image of the model architecture
st.markdown(image_html, unsafe_allow_html=True)

# Describe the model architecture
st.markdown("The model integrates multimodal features from both the speaker and the listener videos. For the speaker, PIXIE extracts facial and upper body motion features from the input video, while Librosa converts the raw audio into mel-spectrogram features. These features are combined using a cross-modal transformer to capture long-range dependencies.")

st.markdown("For the listener, we process facial and upper body past motion sequences along with a personality score. The personality score is encoded by a Personality Network into a conditioned vector, which is then concatenated with a quantized motion sequence obtained from a VQ-VAE. This VQ-VAE has learned a discrete latent space for non-verbal cues from the listener motion.")

st.markdown("The concatenated listener features and personality vector are fed into a predictor transformer, which generates the future motion sequences for the listener, including both facial and upper body movements. This prediction is based on the combined multimodal inputs from the speaker and the listener personality traits.")

st.markdown("Finally, the predicted sequence is matched to the closest codebook entry, and the quantized sequence is decoded to produce the final output. This output represents a personality-aware response from the listener, reflecting the multimodal inputs from the speaker.")


# Describe the training process
st.markdown("""
**Training Process**

We train our model in two stages. First, the VQ-VAE is trained to learn a discrete latent space for motion sequences, creating a learned codebook. In the second stage, the transformers and personality network are trained to generate personalized responses based on the speaker's audio and motion data, along with the listener's personality traits.

Training involves teacher-forcing and random masking to improve autoregressive prediction capabilities. The Adam optimizer is used with a learning rate of 0.001, a batch size of 32, and training occurs over 200 epochs. The model's performance is evaluated based on validation loss, and the best model is selected accordingly.
""")

# Describe the testing process
st.markdown("""
**Testing Process**

During testing, we input zero values for time steps where prior listener predictions are unavailable. Masking is adjusted to support autoregressive predictions for varying input lengths, without relying on ground truth past listener motion data.
""")

st.markdown("""
---

<sub>Learn more about VQ-VAE and see code [here](https://paperswithcode.com/method/vq-vae).</sub>
""", unsafe_allow_html=True)

# Evaluation
st.markdown("<a name='evaluation'></a>", unsafe_allow_html=True)
st.header("Evaluation")
st.markdown("""We evaluated our model using a large dyadic dataset, UDIVA, and conducted extensive experiments to assess its performance. Our evaluation includes quantitative results, a user study, and qualitative analysis. 
We compared our model to a baseline model and demonstrated that it outperforms the baseline in both quantitative and qualitative measures, providing a more nuanced and realistic representation of listener behavior in dyadic interactions. 
Our model is the first of its kind to generate a listener avatar that incorporates the entire upper body, including the face, hands, and body posture, while also considering the listener's personality traits.""")

st.markdown("""
---

<sub>UDIVA dataset description [here](https://chalearnlap.cvc.uab.es/dataset/41/description/).</sub>
""", unsafe_allow_html=True)

st.subheader("➜ Quantitative Results")
st.markdown("""To measure quantitative performance, we use four metrics commonly used to evaluate nonverbal behavior generation approaches. 
We use L2 Distance to directly measure the fit to the ground truth motion. In addition, we use the Frechet Inception Distance (FID) to 
measure the distributional similarity between the generated sequences and the actual ground-truth listener motion sequences present in our dataset.
 We also employ the Paired Frechet Inception Distance (FID) to evaluate the plausibility of the joint speaker-listener motions. 
 Finally, we measure Variance of the generated motion sequence as an indicator of the variability present in the generations.
  A good result is achieved by low L2 Distance, low FID and P-FID, and high variability of the generated motions.

Here are the results of our quantitative evaluations: """)

table_html = """
<style>
table.custom-table {
  width: 100%;
  background-color: #222;
  color: #fff;
  border-collapse: collapse;
  font-family: Arial, sans-serif;
  font-size: 14px;
}
table.custom-table th, table.custom-table td {
  padding: 10px;
  border: 1px solid #555;
}
table.custom-table th {
  background-color: #333;
  text-align: center;
  border-bottom: 2px solid #555;
}
table.custom-table tbody tr:nth-child(odd) {
  background-color: #2a2a2a;
}
table.custom-table tbody tr:nth-child(even) {
  background-color: #1e1e1e;
}
table.custom-table tbody tr:hover {
  background-color: #444;
}
table.custom-table th[colspan="4"] {
  text-align: center;
}
table.custom-table th[colspan="4"] {
  border-bottom: 2px solid #fff;
}
</style>

<table class="custom-table">
  <thead>
    <tr>
      <th></th>
      <th colspan="4">Face (expression & jaw)</th>
      <th colspan="4">Upper Body (body & hand)</th>
    </tr>
    <tr>
      <th>Model</th>
      <th>L2 &darr;</th>
      <th>FID &darr;</th>
      <th>P-FID &darr;</th>
      <th>Var &uarr;</th>
      <th>L2 &darr;</th>
      <th>FID &darr;</th>
      <th>P-FID &darr;</th>
      <th>Var &uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Personality-agnostic Baseline</td>
      <td>32.45</td>
      <td>7.67</td>
      <td>10.47</td>
      <td>1.39</td>
      <td>75.29</td>
      <td>58.87</td>
      <td>96.82</td>
      <td>0.97</td>
    </tr>
    <tr>
      <td>Ours - Joint Body/Face Representation</td>
      <td>33.05</td>
      <td>8.65</td>
      <td>11.94</td>
      <td>1.49</td>
      <td>74.41</td>
      <td>51.90</td>
      <td>91.49</td>
      <td>0.84</td>
    </tr>
    <tr>
      <td>Ours - Random Extraversion Scores</td>
      <td>32.76</td>
      <td>7.58</td>
      <td>10.83</td>
      <td><b>1.61</b></td>
      <td>73.23</td>
      <td>47.56</td>
      <td>91.33</td>
      <td><b>1.13</b></td>
    </tr>
    <tr>
      <td><b>Ours</b></td>
      <td><b>32.12</b></td>
      <td><b>6.15</b></td>
      <td><b>10.31</b></td>
      <td>1.54</td>
      <td><b>72.26</b></td>
      <td><b>43.16</b></td>
      <td><b>87.73</b></td>
      <td>1.03</td>
    </tr>
  </tbody>
</table>
"""

st.write(table_html, unsafe_allow_html=True)

st.subheader("➜ User Study")
st.markdown("""We conducted two experiments using Google Forms to evaluate our final personality-aware model's ability to generate distinguishable listener behaviors based on extraversion personality traits. In the first experiment, we generated two listener avatars using the highest and lowest extraversion scores from our dataset and asked participants to identify which one exhibited more extraverted behavior. 

In the second experiment, we compared our personality-aware model to base model or personality-agnostic model by presenting participants with videos featuring two listener avatars alongside the speaker and asking them to identify which one seemed more engaged with the speaker.

We randomly selected 6 listeners from the test set for each experiment and employed the highest and lowest extraversion scores from our dataset to condition each of the 6 listeners. For the second experiment, we used each of the 6 randomly selected listeners as inputs for both models. We also randomized the positions of the two listener avatars in each video to prevent any bias.

The study involved 20 participants, and the results showed that our personality-aware model was capable of accurately differentiating behaviors between introverted and extroverted listeners in 86% of the cases. Moreover, in 71% of the cases, participants showed a preference for our personality-aware model over the personality-agnostic/base model when utilizing actual extraversion scores.""")

# Load the image
import plotly.graph_objects as go

fig = go.Figure(data=[
 go.Bar(name='Preference for Personality-Model II', x=['Preference'], y=[71]),
 go.Bar(name='Distinguishing Extraversion', x=['Distinguishing'], y=[86])
])
fig.update_layout(barmode='group', xaxis_title='', yaxis_title='Percentage')
st.plotly_chart(fig)

st.subheader("➜ Qualitative Results")
st.markdown("""Our model results, as depicted in the figure below, present a comparative analysis of the speaker's frame sequence with the generated introverted and extroverted listener avatar sequences. The top row of the figure showcases the introverted listener, characterized by minimal facial expressions, limited eye contact, and restrained body language. In contrast, the middle row features the extroverted listener, who demonstrates more active engagement through frequent smiling, leaning towards the speaker, and dynamic hand gestures. The bottom row represents the speaker's frame sequence. This qualitative result highlights the model's ability to discern and generate listener responses that align with distinct personality traits, with the extroverted avatar displaying more pronounced engagement cues towards the speaker.""")
# Load the image
model_architecture = Image.open('Images/Qualttive-Results.png')

# Convert the image to base64
buffer = BytesIO()
model_architecture.save(buffer, format='PNG')
img_str = base64.b64encode(buffer.getvalue()).decode()

# Create the HTML string for the image container
image_html = f"""
<div class="image-container">
    <img src="data:image/png;base64,{img_str}" alt="Model Architecture">
    <div style="font-size: 14px; margin-top: 8px; color: grey; text-align: center;"> Visualizing our model results (Extrovert vs. Introvert different behavior generations)</div>
    <div class="enlarged-image">
        <img src="data:image/png;base64,{img_str}" alt="Enlarged Model Architecture">
    </div>
</div>
"""

# Display the image container
st.markdown(image_html, unsafe_allow_html=True)



# Implementation
st.markdown("<a name='implementation'></a>", unsafe_allow_html=True)

st.header("Implementation")
st.markdown("""
Our project was developed using Python and PyTorch, leveraging a diverse set of tools and libraries to achieve efficient and effective machine learning outcomes.

**Slurm Workload Manager** was employed to handle computational resources, utilizing its Job Arrays feature to run multiple experiments in parallel. This approach optimized resource usage and significantly reduced overall runtime.

**Enroot images** were used for containerization, providing a consistent and isolated environment similar to Docker containers. These images were deployed on Slurm cluster GPU partitions, ensuring a stable and reproducible setup for our experiments.

**NVIDIA A100-40GB GPUs**: We used eight of these powerful GPUs for model training, enabling parallel processing and rapid computation required for large-scale models.
""")

st.markdown("### Tools and Libraries")

with st.expander("Click to show used tools and libraries", expanded=False):
    st.markdown("""
    - PyTorch
    - TorchVision
    - NumPy
    - SciPy
    - Pandas
    - TensorBoard
    - OpenCV
    - Librosa
    - PIL (Pillow)
    - Scikit-learn
    - Multiprocessing
    - Joblib
    - TQDM
    - JSON
    - Logging
    - Argparse
    - Shutil
    - Subprocess
    - Pickle
    - Datetime
    - Math
    - SciPy.io
    """)

st.markdown("""
The following sub-sections will detail our implementation process and provide code snippets illustrating the use of these tools and libraries.
""")


# Data Preprocessing
st.subheader("Data Preprocessing")
st.markdown("""
In this step, we preprocessed our dataset to make it suitable for model training. The main tasks included:

1. **Dyadic Video Segmentation**: We loaded dyadic videos and segmented them into separate speaker and listener streams for each participant using their corresponding transcripts with timestamps. The data was processed in parallel by loading subdirectories of videos into batches using Python's multiprocessing.
""")

# Dyadic Video Segmentation Code Snippet
with st.expander("Dyadic Video Segmentation Code"):
    st.code("""
    import os
    import sys
    from tqdm import tqdm
    import re
    from moviepy.editor import *
    import multiprocessing
    import logging

    logging.basicConfig(filename='video_sync_error.log', level=logging.ERROR)

    def extract_timestamps(transcript_file):
        timestamps = {}
        with open(transcript_file, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                line = line.strip()
                match = re.match("(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)", line)
                if match:
                    start_time = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3)) + int(
                        match.group(4)) / 1000
                    end_time = int(match.group(5)) * 3600 + int(match.group(6)) * 60 + int(match.group(7)) + int(
                        match.group(8)) / 1000
                    duration = end_time - start_time
                    if duration >= 1:
                        if i + 1 < len(lines):
                            speaker = lines[i + 1].strip().split(": ")[0]
                            if speaker not in timestamps:
                                timestamps[speaker] = []
                            timestamps[speaker].append((start_time, end_time))
                i += 1
        return timestamps

    def process_subdir(subdir, recordings_dir, transcriptions_dir, output_dir):
        try:
            recording_subdir = os.path.join(recordings_dir, subdir)
            transcription_subdir = os.path.join(transcriptions_dir, subdir)
            transcript_file = os.path.join(transcription_subdir, subdir + "_talk.srt")
            video_file1 = os.path.join(recording_subdir, "FC1_T.mp4")
            video_file2 = os.path.join(recording_subdir, "FC2_T.mp4")

            timestamps = extract_timestamps(transcript_file)
            part1_timestamps = [(start_time, end_time) for start_time, end_time in timestamps.get("PART.1", [])]
            part2_timestamps = [(start_time, end_time) for start_time, end_time in timestamps.get("PART.2", [])]

            sub_output_dir = os.path.join(output_dir, "train", "video_sync", subdir)
            if not os.path.exists(sub_output_dir):
                os.makedirs(sub_output_dir)

            with VideoFileClip(video_file1) as clip1:
                part1_clips = [clip1.subclip(start_time, min(end_time, clip1.duration)) for start_time, end_time in
                               part1_timestamps]
                part1_clips = concatenate_videoclips(part1_clips)
                part1_only_listening_clips = concatenate_videoclips(
                    [clip1.subclip(start_time, min(end_time, clip1.duration)) for start_time, end_time in
                     part2_timestamps])

                part1_clips.write_videofile(os.path.join(sub_output_dir, "part1_speak_all_body_pixie.mp4"),
                                            temp_audiofile=os.path.join(sub_output_dir, "part1_speak_all_body_pixie.mp3"),
                                            remove_temp=False)
                part1_only_listening_clips.write_videofile(
                    os.path.join(sub_output_dir, "part1_list_all_body_pixie.mp4"),
                    temp_audiofile=os.path.join(sub_output_dir, "part1_list_all_body_pixie.mp3"),
                    remove_temp=True)

            with VideoFileClip(video_file2) as clip2:
                part2_clips = [clip2.subclip(start_time, min(end_time, clip2.duration)) for start_time, end_time in
                               part2_timestamps]
                part2_clips = concatenate_videoclips(part2_clips)
                part2_only_listening_clips = concatenate_videoclips(
                    [clip2.subclip(start_time, min(end_time, clip2.duration)) for start_time, end_time in
                     part1_timestamps])

                part2_clips.write_videofile(os.path.join(sub_output_dir, "part2_speak_all_body_pixie.mp4"),
                                            temp_audiofile=os.path.join(sub_output_dir, "part2_speak_all_body_pixie.mp3"),
                                            remove_temp=False)
                part2_only_listening_clips.write_videofile(
                    os.path.join(sub_output_dir, "part2_list_all_body_pixie.mp4"),
                    temp_audiofile=os.path.join(sub_output_dir, "part2_list_all_body_pixie.mp3"),
                    remove_temp=True)
        except Exception as e:
            logging.exception(f"An error occurred while processing subdirectory {subdir}: {e}")

    def process_batch(batch, recordings_dir, transcriptions_dir, output_dir):
        try:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.starmap(process_subdir, [(subdir, recordings_dir, transcriptions_dir, output_dir) for subdir in batch])
        except Exception as e:
            logging.exception(f"An error occurred while processing batch {batch}: {e}")

    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print("Usage: python video_sync.py <input_dir> <output_dir>")
            sys.exit(1)

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]

        recordings_dir = os.path.join(input_dir, "train/recordings")
        transcriptions_dir = os.path.join(input_dir, "train/transcriptions")

        # Read subdirectories
        subdirs = [d for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))]

        # Set the number of batches and number of processes
        batch_size = 11  # number of subdirectories to process in each batch

        num_batches = len(subdirs) // batch_size + 1

        # Divide the subdirs into batches
        batches = [subdirs[i:i + batch_size] for i in range(0, len(subdirs), batch_size)]

        print(f"Number of subdirectories {len(subdirs)}, processed in {num_batches} batches, with batch size {batch_size}")

        # Process batches
        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1} of {len(batches)}")
            process_batch(batch, recordings_dir, transcriptions_dir, output_dir)
    """, language='python')

st.markdown("""
2. **Cropping Video Frames:** We cropped both speaker and listener videos frames around each participant using the corresponding dataset annotations for each participant.  
""")
with st.expander("Cropping Video Frames Code"):
    st.code("""
    import os
    import sys
    from tqdm import tqdm
    from moviepy.editor import *
    import multiprocessing
    import logging
    import cv2
    import argparse
    import h5py
    import zipfile
    import numpy as np
    import io

    logging.basicConfig(filename='video_crop_error.log', level=logging.ERROR)

    def crop_body_from_video(avg_landmarks, frame):
        avg_landmarks = avg_landmarks[:6, :2]

        # Get the x and y coordinates of the leftmost, rightmost, topmost and bottommost landmarks
        x_left = int(min(avg_landmarks[:, 0]))
        x_right = int(max(avg_landmarks[:, 0]))
        y_top = int(min(avg_landmarks[:, 1]))
        y_bottom = int(max(avg_landmarks[:, 1]))

        # Add 65% to each side of the bounding box
        # Add 100% to top and bottom of the bounding box
        width = x_right - x_left
        height = y_bottom - y_top
        x_left -= int(0.65 * width)
        x_right += int(0.65 * width)
        y_top -= int(1.00 * height)
        y_bottom += int(1.00 * height)

        # Get the frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Validate the bounding box coordinates
        x_left = max(0, x_left)
        x_right = min(frame_width, x_right)
        y_top = 35
        y_bottom = frame_height

        # Crop the frame to the bounding box
        cropped_frame = frame[y_top:y_bottom, x_left:x_right]

        # Resize the cropped frame to the target size
        target_size = (820, 935)
        resized_frame = cv2.resize(cropped_frame, target_size)

        return resized_frame

    def draw_bounding_box_on_video(avg_landmarks, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        avg_landmarks = avg_landmarks[:6, :2]

        # Get the x and y coordinates of the leftmost, rightmost, topmost and bottommost landmarks
        x_left = int(min(avg_landmarks[:, 0]))
        x_right = int(max(avg_landmarks[:, 0]))
        y_top = int(min(avg_landmarks[:, 1]))
        y_bottom = int(max(avg_landmarks[:, 1]))

        # Add 65% to each side of the bounding box
        width = x_right - x_left
        height = y_bottom - y_top
        x_left -= int(0.65 * width)
        x_right += int(0.65 * width)
        y_top -= int(1.00 * height)
        y_bottom += int(1.00 * height)

        # Draw bounding box on frame
        cv2.rectangle(frame_rgb, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)

        # Convert frame back to BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        return frame_bgr

    def get_average_landmarks(landmarks):
        avg_landmarks = {}
        for part in landmarks.keys():
            if landmarks[part]:
                avg_landmarks[part] = np.mean(landmarks[part], axis=0)
        return avg_landmarks

    def filter_body_landmarks(l):
        body_joints = [9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        return l[body_joints]

    def extract_landmarks(path_to_zip_1, path_to_zip_2, all=False):
        with zipfile.ZipFile(path_to_zip_1) as zip1, zipfile.ZipFile(path_to_zip_2) as zip2:
            zip1_file = zip1.namelist()[0]
            zip2_file = zip2.namelist()[0]
            with io.BytesIO(zip1.read(zip1_file)) as f1, io.BytesIO(zip2.read(zip2_file)) as f2:
                landmarks = {"PART.1": [], "PART.2": []}
                with h5py.File(f1, "r") as hdf5_1, h5py.File(f2, "r") as hdf5_2:
                    for key_frame in hdf5_1.keys():
                        if "body" not in hdf5_1[key_frame]:
                            continue
                        landmarks_1 = hdf5_1[key_frame]["body"]["landmarks"][()] if "landmarks" in hdf5_1[key_frame]["body"].keys() else None
                        if not all and landmarks_1 is not None:
                            landmarks_1 = filter_body_landmarks(landmarks_1).astype(int)
                        landmarks["PART.1"].append(landmarks_1)

                    for key_frame in hdf5_2.keys():
                        if "body" not in hdf5_2[key_frame]:
                            continue
                        landmarks_2 = hdf5_2[key_frame]["body"]["landmarks"][()] if "landmarks" in hdf5_2[key_frame]["body"].keys() else None
                        if not all and landmarks_2 is not None:
                            landmarks_2 = filter_body_landmarks(landmarks_2).astype(int)
                        landmarks["PART.2"].append(landmarks_2)
        return landmarks

    def process_subdir(subdir, video_sync_dir, annotations_dir, output_dir):
        try:
            video_sync_subdir = os.path.join(video_sync_dir, subdir)
            annotations_dir_subdir = os.path.join(annotations_dir, subdir)
            annotations_file_part1 = os.path.join(annotations_dir_subdir, "FC1_T", "annotations_raw_unmasked.zip")
            annotations_file_part2 = os.path.join(annotations_dir_subdir, "FC2_T", "annotations_raw_unmasked.zip")

            video_file1 = os.path.join(video_sync_subdir, "part1_speak_all_body_pixie.mp4")
            video_file2 = os.path.join(video_sync_subdir, "part1_list_all_body_pixie.mp4")
            video_file3 = os.path.join(video_sync_subdir, "part2_speak_all_body_pixie.mp4")
            video_file4 = os.path.join(video_sync_subdir, "part2_list_all_body_pixie.mp4")

            landmarks = extract_landmarks(annotations_file_part1, annotations_file_part2, all=False)
            avg_landmarks = get_average_landmarks(landmarks)

            sub_output_dir = os.path.join(output_dir, "train", "video_crop", subdir)
            if not os.path.exists(sub_output_dir):
                os.makedirs(sub_output_dir)

            with VideoFileClip(video_file1) as clip1:
                clip1 = clip1.fl_image(lambda frame: crop_body_from_video(avg_landmarks['PART.1'], frame))
                clip1.write_videofile(os.path.join(sub_output_dir, "part1_speak_all_body_pixie.mp4"),
                                      temp_audiofile=os.path.join(sub_output_dir, "part1_speak_all_body_pixie.mp3"),
                                      remove_temp=False)

            with VideoFileClip(video_file2) as clip2:
                clip2 = clip2.fl_image(lambda frame: crop_body_from_video(avg_landmarks['PART.1'], frame))
                clip2.write_videofile(os.path.join(sub_output_dir, "part1_list_all_body_pixie.mp4"),
                                      temp_audiofile=os.path.join(sub_output_dir, "part1_list_all_body_pixie.mp3"),
                                      remove_temp=True)

            with VideoFileClip(video_file3) as clip3:
                clip3 = clip3.fl_image(lambda frame: crop_body_from_video(avg_landmarks['PART.2'], frame))
                clip3.write_videofile(os.path.join(sub_output_dir, "part2_speak_all_body_pixie.mp4"),
                                      temp_audiofile=os.path.join(sub_output_dir, "part2_speak_all_body_pixie.mp3"),
                                      remove_temp=False)

            with VideoFileClip(video_file4) as clip4:
                clip4 = clip4.fl_image(lambda frame: crop_body_from_video(avg_landmarks['PART.2'], frame))
                clip4.write_videofile(os.path.join(sub_output_dir, "part2_list_all_body_pixie.mp4"),
                                      temp_audiofile=os.path.join(sub_output_dir, "part2_list_all_body_pixie.mp3"),
                                      remove_temp=True)

        except Exception as e:
            logging.exception(f"An error occurred while processing subdirectory {subdir}: {e}")

    def process_batch(batch, video_sync_dir, annotations_dir, output_dir):
        try:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.starmap(process_subdir, [(subdir, video_sync_dir, annotations_dir, output_dir) for subdir in batch])
        except Exception as e:
            logging.exception(f"An error occurred while processing batch {batch}: {e}")

    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print("Usage: python crop.py <input_dir> <output_dir>")
            sys.exit(1)

        video_sync_dir = sys.argv[1]
        output_dir = sys.argv[2]

        annotations_dir = "/home/FAREWELL-CHANNEL_2/ANNO/"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        subdirs = os.listdir(video_sync_dir)
        subdirs = sorted([s for s in subdirs if os.path.isdir(os.path.join(video_sync_dir, s))])

        batches = [subdirs[i:i + 10] for i in range(0, len(subdirs), 10)]

        for batch in tqdm(batches):
            process_batch(batch, video_sync_dir, annotations_dir, output_dir)
    """, language="python")

st.markdown("""
3. **Extracting Motion Features:** We used [PIXIE](https://github.com/yfeng95/PIXIE) to process both speaker and listener videos.  
This allows us to extract SMPL-X parameters for each video frame, including the 3D body shape, pose, hand articulation, and facial expressions.
""")

# Extracting Motion Features Code Snippet
with st.expander("Extracting Motion Features Code"):
    st.code("""
    import argparse
    import cv2
    import numpy as np
    import os
    import pickle
    import sys
    import pdb
    import torch
    import torch.backends.cudnn as cudnn
    import multiprocessing as mp
    import logging
    import traceback
    from tqdm import tqdm

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pixielib.pixie import PIXIE
    from pixielib.datasets.body_datasets import TestData
    from pixielib.utils import util
    from pixielib.utils.config import cfg as pixie_cfg
    from torch.utils.data import Dataset, DataLoader

    logging.basicConfig(filename='pixie_error.log', level=logging.ERROR, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


    def process_subdir(pixie, subdir, frames_crop_dir, output_dir, device):
        try:
            torch.cuda.set_device(device)
            # visualizer = Visualizer(render_size=args.render_size, config = pixie_cfg, device=device, rasterizer_type=args.rasterizer_type)
            sub_dir_path = os.path.join(frames_crop_dir, subdir)
            video_folders = [os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) if
                            os.path.isdir(os.path.join(sub_dir_path, f))]
            output_subdir = os.path.join(output_dir, "train", "encode_pixie_id", subdir)
            # Check if the output subdirectory exists
            if not os.path.exists(output_subdir):
                logging.info(f"Start processing subdirectory {subdir}")
                print(f"Start processing subdirectory {subdir}")
                for video_folder in video_folders:
                    testdata = TestData(video_folder, iscrop=False, body_detector='rcnn', device=device)
                    # pdb.set_trace()
                    # Run PIXIE
                    for i, frames_batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
                        util.move_dict_to_device(frames_batch, device)
                        # pdb.set_trace()
                        frames_batch['image'] = frames_batch['image'].unsqueeze(0)
                        frames_batch['image_hd'] = frames_batch['image_hd'].unsqueeze(0)
                        name = frames_batch['name']

                        data = {
                            'body': frames_batch
                        }

                        param_dict = pixie.encode(data)
                        codedict = param_dict['body']

                        video_name = os.path.basename(video_folder[:-1])
                        os.makedirs(os.path.join(output_subdir, video_name), exist_ok=True)
                        util.save_pkl(
                            os.path.join(output_subdir, video_name, f"{name}_param.pkl"),
                            codedict)
                        # Deleting tensors and freeing up GPU memory
                        del frames_batch, data, param_dict, codedict
                        torch.cuda.empty_cache()
                logging.info(f"Finished processing subdirectory {subdir}")
                print(f"Finished processing subdirectory {subdir}")
            else:
                logging.info(f"Subdirectory {subdir} already exists in the output directory {output_dir}")
                print(f"Subdirectory {subdir} already exists in the output directory {output_dir}")
                for video_folder in video_folders:
                    video_name = os.path.basename(video_folder[:-1])
                    video_dir = os.path.join(output_subdir,video_name)
                    if not os.path.exists(video_dir):
                        print(f"Video directory {video_dir} doesn't exist in the output sub-directory {subdir}")
                        testdata = TestData(video_folder, iscrop=False, body_detector='rcnn', device=device)
                        # pdb.set_trace()
                        # Run PIXIE
                        for i, frames_batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
                            util.move_dict_to_device(frames_batch, device)
                            # pdb.set_trace()
                            frames_batch['image'] = frames_batch['image'].unsqueeze(0)
                            frames_batch['image_hd'] = frames_batch['image_hd'].unsqueeze(0)
                            name = frames_batch['name']
                            data = {
                                'body': frames_batch
                                }
                            param_dict = pixie.encode(data)
                            codedict = param_dict['body']
                            video_name = os.path.basename(video_folder[:-1])
                            os.makedirs(os.path.join(output_subdir, video_name), exist_ok=True)
                            util.save_pkl(
                                os.path.join(output_subdir, video_name, f"{name}_param.pkl"),
                                codedict)
                            # Deleting tensors and freeing up GPU memory
                            del frames_batch, data, param_dict, codedict
                            torch.cuda.empty_cache()
                    else:
                        if len(os.listdir(video_dir)) != len(os.listdir(video_folder)):
                            print(f"adding more files in {video_dir}")
                            testdata = TestData(video_folder, iscrop=False, body_detector='rcnn', device=device)
                            # pdb.set_trace()
                            # Run PIXIE
                            for i, frames_batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
                                util.move_dict_to_device(frames_batch, device)
                                # pdb.set_trace()
                                frames_batch['image'] = frames_batch['image'].unsqueeze(0)
                                frames_batch['image_hd'] = frames_batch['image_hd'].unsqueeze(0)
                                name = frames_batch['name']
                                data = {
                                    'body': frames_batch
                                    }
                                param_dict = pixie.encode(data)
                                codedict = param_dict['body']
                                util.save_pkl(
                                    os.path.join(video_dir, f"{name}_param.pkl"),
                                    codedict)
                                # Deleting tensors and freeing up GPU memory
                                del frames_batch, data, param_dict, codedict
                                torch.cuda.empty_cache()
                logging.info(f"Finished processing subdirectory {subdir}")
                print(f"Finished processing subdirectory {subdir}")

        except Exception as e:
            logging.exception(f"An error occurred while processing subdirectory {subdir}: {e}")
            print(f"An error occurred while processing subdirectory {subdir}: {e}")
            logging.error(traceback.format_exc())  # write the traceback to the log file
            traceback.print_exc(file=sys.stdout)  # print the traceback to the console


    def process_batch(batch, frames_crop_dir, output_dir, devices, num_gpus):
        try:
            logging.info(f"Start processing batch {batch}")
            print(f"Start processing batch {batch}")
            # Map subdirectories to devices
            subdir_to_device = {subdir: devices[i % len(devices)] for i, subdir in enumerate(batch)}
            # Create PIXIE instance for each GPU
            pixies = [PIXIE(config=pixie_cfg, device=device) for device in devices]
            # Map the subdirectories to the corresponding PIXIE instance
            subdir_to_pixie = {subdir: pixies[i % len(pixies)] for i, subdir in enumerate(batch)}
            with mp.Pool(processes=num_gpus*4) as pool:
                pool.starmap(process_subdir,
                            [(subdir_to_pixie[subdir], subdir, frames_crop_dir, output_dir,
                            subdir_to_device[subdir]) for
                            subdir in
                            batch])
            # Delete the PIXIE instances
            for pixie_instance in pixies:
                del pixie_instance
            torch.cuda.empty_cache()
            logging.info(f"Finished processing batch {batch}")
            print(f"Finished processing batch {batch}")
        except Exception as e:
            logging.exception(f"An error occurred while processing batch {batch}: {e}")
            print(f"An error occurred while processing batch {batch}: {e}")
            logging.error(traceback.format_exc())  # write the traceback to the log file
            traceback.print_exc(file=sys.stdout)  # print the traceback to the console


    class MyDataset(Dataset):
        def __init__(self, frames_crop_dir):
            super().__init__()
            self.video_folders = []
            for subdir in os.listdir(frames_crop_dir):
                sub_dir_path = os.path.join(frames_crop_dir, subdir)
                self.video_folders.extend([os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) if
                                        os.path.isdir(os.path.join(sub_dir_path, f))])
            self.data = {}
            for video_folder in self.video_folders:
                subdir = os.path.basename(os.path.dirname(video_folder))
                if subdir not in self.data:
                    self.data[subdir] = []
                testdata = TestData(video_folder, iscrop=True, body_detector='rcnn')
                for frames_batch in testdata:
                    self.data[subdir].append((frames_batch, os.path.basename(video_folder)))


    if __name__ == "__main__":
        # Set the multiprocessing start method to 'spawn'
        mp.set_start_method('spawn', force=True)
        if len(sys.argv) != 3:
            print("Usage: python demo_fit_body_mult_gpu.py <input_dir> <output_dir> <device>")
            sys.exit(1)

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]

        # check env
        if not torch.cuda.is_available():
            print('CUDA is not available! use CPU instead')
        else:
            cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]
            print("Number of used GPUS are ",num_gpus)
            print(devices)

        frames_crop_dir = os.path.join(input_dir, "extracted_features/train/frames_mirror_id")

        # my_dataset = MyDataset(frames_crop_dir)
        # subdir_data = my_dataset.data['subdir1']

        # Read subdirectories
        sub_dirs = [d for d in os.listdir(frames_crop_dir) if os.path.isdir(os.path.join(frames_crop_dir, d))]

        # Split list of subdirectories into batches
        batch_size = 15  # number of subdirectories to process in each batch
        num_batches = 1

        # Divide the subdirs into batches
        subdir_batches = [sub_dirs[i:i + batch_size] for i in range(0, len(sub_dirs), batch_size)]

        print(f"Number of subdirectories {len(sub_dirs)}, processed in {num_batches} batches, with batch size {batch_size}")

        # Create a pool of processes and map the batches to the processes
        for i, batch in enumerate(subdir_batches):
            print(f"Processing batch {i + 1}/{num_batches}")
            process_batch(batch, frames_crop_dir, output_dir, devices, num_gpus)

        logging.info("All batches are processed")
        """, language="python")


st.markdown("""
4. **Extracting Audio Features:** We extracted audio features using Mel-spectrograms via librosa, with parameters such as a 16 kHz sample rate, a 25 ms window length, and a 10 ms hop length.
The features were resized to match the video frame count, organized into a 3D array, and saved as .npy files for each audio file.
""")

with st.expander("Extracting Audio Features Code"):
    st.code("""
    import os
    import sys
    import argparse
    import pdb
    from tqdm import tqdm
    import multiprocessing
    import logging
    import traceback
    import numpy as np
    import cv2
    import glob
    import librosa
    import numpy as np
    from PIL import Image

    logging.basicConfig(filename='audio_final_error.log', level=logging.ERROR,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


    def load_mfcc(audio_path, num_frames_video):
        waveform, sample_rate = librosa.load('{}'.format(audio_path), sr=16000)
        win_len = int(0.025 * sample_rate)
        hop_len = int(0.010 * sample_rate)
        fft_len = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
        S_dB = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, hop_length=hop_len)

        # do some resizing to match frame rate
        im = Image.fromarray(S_dB)
        T = 64
        _, feature_dim = im.size
        scale_four = T * 4
        # Reshape im to be (N,scale_four,128)
        N = num_frames_video // 64
        # Convert the image to a numpy array
        im = np.transpose(np.asarray(im))
        S_dB = im[:N * scale_four, :]
        S_dB = S_dB.reshape(N, scale_four, 128)
        # print(S_dB.shape)
        return S_dB


    def process_subdir(subdir, video_sync_dir, frames_crop_dir, output_dir):
        try:
            sub_dir_path = os.path.join(video_sync_dir, subdir)
            output_subdir = os.path.join(output_dir, "test", "audio_features_session", subdir)
            # Check if the output subdirectory exists
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir, exist_ok=True)
            # use glob to find all the .mp3 files in the subdir
            audio_files = glob.glob(sub_dir_path + '/*.mp3')
            # pdb.set_trace()
            # loop over the audio files and process each one
            for audio_file in tqdm(audio_files):
                # extract the file name without extension from the audio file path
                audio_name = os.path.splitext(os.path.basename(audio_file))[0]

                # construct the path to the corresponding video file
                video_folder = os.path.join(os.path.join(frames_crop_dir, subdir), audio_name)
                num_frames_video = len(os.listdir(video_folder))
                # print(audio_file, num_frames_video)
                # pdb.set_trace()
                # load_mfcc function on the audio file
                mfcc = load_mfcc(audio_file, num_frames_video)
                audio_name = audio_name.replace("_all_body_pixie", "_audio_mfcc")
                np.save(os.path.join(output_subdir, audio_name + '.npy'), mfcc)
            logging.info(f"Finished processing subdirectory {subdir}")
            print(f"Finished processing subdirectory {subdir}")
        except Exception as e:
            logging.exception(f"An error occurred while processing subdirectory {subdir}: {e}")
            print(f"An error occurred while processing subdirectory {subdir}: {e}")
            logging.error(traceback.format_exc())  # write the traceback to the log file
            traceback.print_exc(file=sys.stdout)  # print the traceback to the console


    def process_batch(batch, video_sync_dir, frames_crop_dir, output_dir):
        try:
            logging.info(f"Start processing batch {batch}")
            print(f"Start processing batch {batch}")
            with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 6) as pool:
                pool.starmap(process_subdir, [(subdir, video_sync_dir, frames_crop_dir, output_dir) for subdir in batch])
            logging.info(f"Finished processing batch {batch}")
            print(f"Finished processing batch {batch}")
        except Exception as e:
            logging.exception(f"An error occurred while processing batch {batch}: {e}")
            print(f"An error occurred while processing batch {batch}: {e}")
            logging.error(traceback.format_exc())  # write the traceback to the log file
            traceback.print_exc(file=sys.stdout)  # print the traceback to the console


    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print("Usage: python saving_audio_final.py <input_dir> <output_dir>")
            sys.exit(1)

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]

        video_sync_dir = os.path.join(input_dir, "extracted_features/test/video_sync")
        frames_crop_dir = os.path.join(input_dir, "extracted_features/test/frames_crop")

        # Read subdirectories
        subdirs = [d for d in os.listdir(video_sync_dir) if os.path.isdir(os.path.join(video_sync_dir, d))]

        # Set the number of batches and number of processes
        batch_size = 11  # number of subdirectories to process in each batch
        # num_processes_per_batch = 2  # number of processes to use for each batch

        num_batches = len(subdirs) // batch_size + 1

        # Divide the subdirs into batches
        batches = [subdirs[i:i + batch_size] for i in range(0, len(subdirs), batch_size)]

        print(f"Number of subdirectories {len(subdirs)}, processed in {num_batches} batches, with batch size {batch_size}")

        # Create a pool of processes and map the batches to the processes
        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1} of {len(batches)}")
            process_batch(batch, video_sync_dir, frames_crop_dir, output_dir)

        logging.info("All batches are processed")
        print("All batches are processed")
        """, language="python")

st.markdown("""
5. **Postprocessing (motion features):** We applied a dual filtering strategy to smooth the motion features and reduce jittering. First, a mean average filter with a window size of 7 frame was used to smooth transitions between frames.
Then, a median filter with a 3-frame window was applied at detected scene changes to minimize abrupt jumps while preserving motion variability.
""")

with st.expander("Postprocessing Features Code"):
    st.code("""
    import os, sys
    import numpy as np
    import pickle
    import torch
    from multiprocessing import Pool
    from tqdm import tqdm
    import pdb
    import cv2


    def load_pickle_file(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def save_to_target_directory(updated_content, source_path, source_directory, target_directory):
        target_path = source_path.replace(source_directory, target_directory)
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target_path, 'wb') as f:
            pickle.dump(updated_content, f)

    def calculate_optical_flow(prev_frame, current_frame):
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                motion = np.sqrt((good_new[:,0] - good_old[:,0])**2 + (good_new[:,1] - good_old[:,1])**2)
            else:
                motion = np.array([0])
        except Exception as e:
            print(f"Error in calculate_optical_flow: {e}")
            motion = np.array([0])

        return motion

    def detect_scene_changes(frame_paths, threshold=6.0):
        \"\"\"Detects scene changes in a list of frames using optical flow.\"\"\"
        prev_frame = cv2.imread(frame_paths[0])
        scene_changes = []

        for i, frame_path in enumerate(frame_paths[1:], 1):
            current_frame = cv2.imread(frame_path)
            motion = calculate_optical_flow(prev_frame, current_frame)
            # print(f"frame {i}:  {np.mean(motion)}")
            # Check if the average motion is above the threshold
            if np.mean(motion) > threshold:
                scene_changes.append(i)  # Store the index of the frame

            prev_frame = current_frame

        return scene_changes

    def smooth_predictions(predictions, window=7):
        smoothed = []
        for i in range(len(predictions)):
            if window <= i < len(predictions) - window:  # for middle frames with sufficient surrounding frames
                smoothed_codedict = {}
                # Identify the first parameter (key)
                first_param = next(iter(predictions[i]))
                for param in predictions[i].keys():
                    if param == first_param:
                        # Copy the first parameter as is, without smoothing
                        smoothed_codedict[param] = predictions[i][param]
                    else:
                        # Smooth other parameters
                        sum_frames = sum(predictions[j][param] for j in range(i - window, i + window + 1))
                        smoothed_codedict[param] = sum_frames / (2 * window + 1)
                smoothed.append(smoothed_codedict)
            else:  # for first and last frames where the window is not full
                smoothed.append(predictions[i])
        return smoothed


    def apply_median_filter_at_jumps(predictions, scene_changes, window=3):
        if not scene_changes:
            return predictions

        smoothed_predictions = predictions.copy()
        half_window = window // 2

        for i in scene_changes:
            for param in predictions[0].keys():
                start_index = max(i - half_window, 0)
                end_index = min(i + half_window + 1, len(predictions))
                window_values = [predictions[j][param] for j in range(start_index, end_index)]
                if window_values:
                    array_values = np.array(window_values)
                    median_value = np.median(array_values, axis=0)
                    smoothed_predictions[i][param] = median_value

        return smoothed_predictions

    def process_subdirectory(subdir, source_directory, frames_directory, target_directory):
        try:
            sorted_filenames = sorted([f for f in os.listdir(subdir) if f.endswith("_param.pkl")])

            if not sorted_filenames:
                print(f"No files found in: {subdir}")
                return

            frames_features = [load_pickle_file(os.path.join(subdir, filename)) for filename in sorted_filenames]
            frames_path = subdir.replace(source_directory, frames_directory) + 'e'
            original_frames_list = sorted([os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith('.png') or f.endswith('.jpg')])
            # Detect video cuts that corresponds to abrupt jumps
            print("detecting video cuts.....")
            scene_changes = detect_scene_changes(original_frames_list)
            # Transfer parameters from the first frame to all other frames
            first_frame_parameters = frames_features[0]
            parameters_to_override = ['shape', 'tex', 'body_cam', 'light', 'global_pose']
            for i in range(1, len(frames_features)):
                for param in parameters_to_override:
                    frames_features[i][param] = first_frame_parameters[param]

            print("applying avearge mean filter.....")
            smoothed_features = smooth_predictions(frames_features)
            # Apply median filter at scene change indices
            print("applying median filter....")
            smoothed_features = apply_median_filter_at_jumps(smoothed_features, scene_changes)

            for filename, updated_content in zip(tqdm(sorted_filenames), smoothed_features):
                source_path = os.path.join(subdir, filename)
                save_to_target_directory(updated_content, source_path, source_directory, target_directory)
            print(f"Finished processing subdirectory {subdir}")
        except Exception as e:
            print(f"Error in process_subdirectory for {subdir}: {e}")

    def worker_initializer():
        print(f"Initialized worker {os.getpid()}")

    def main():
        source_directory = "/home/UDIVAv0.5/extracted_features/test/encode_pixie_id"
        frames_directory = "/home/UDIVAv0.5/extracted_features/test/frames_mirror_id"
        target_directory = "/home/UDIVAv0.5/extracted_features/test/smoothed_encode_pixie_id"
        
        top_level_dirs = [os.path.join(source_directory, d) for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
        # import pdb; pdb.set_trace()
        # top_level_dirs = ['/home/UDIVAv0.5/extracted_features/train/encode_pixie_id/20']
        subdirs = ["p0_speak_all_body_pixi", "p0_list_all_body_pixi", "p1_list_all_body_pixi", "p1_speak_all_body_pixi"]
        
        all_subdirs = []
        for top_dir in top_level_dirs:
            for subdir in subdirs:
                path = os.path.join(top_dir, subdir)
                if os.path.exists(path):
                    all_subdirs.append(path)

        with Pool(processes=min(44, os.cpu_count())) as pool:
            pool.starmap(process_subdirectory, [(subdir, source_directory, frames_directory, target_directory) for subdir in all_subdirs])
        
        print("...Finished processing all directories...")

    if __name__ == "__main__":
        main()
""", language="python")

st.markdown("""
6. **Saving Final Features:** Finally, the smoothed motion parameters were resized and saved as updated 3D arrays in the target train directory.
""")

with st.expander("Saving Final Features Code"):
    st.code("""
    import os
    import sys
    import argparse
    import pdb
    from tqdm import tqdm
    import multiprocessing
    import logging
    import traceback
    import numpy as np
    import torch.backends.cudnn as cudnn
    import torch
    import cv2
    import pickle
    import glob
    import shutil

    logging.basicConfig(filename='motion_final_error.log', level=logging.ERROR, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


    def load_data(data_file):
        # Loading the dictionary
        with open(data_file, 'rb') as f:
            features = pickle.load(f)

        # Convert the values to torch tensors
        for key in features.keys():
            features[key] = torch.tensor(features[key])

        # Print the shape of the features
        # for key, value in features.items():
        # print(f"{key} shape: {value.shape}")
        return features


    def prepare_param(features):
        # concatenating frame features and return a tensor with smpl-x parameters.

        features['body_pose'] = torch.cat((features['partbody_pose'], features['neck_pose'], features['head_pose'],
                                        features['left_wrist_pose'], features['right_wrist_pose']), dim=1)
        # note:shape parameters are not assigned to the tensor right now,since it is not use during valing the mode
        # parameters = torch.cat((features['exp'], features['global_pose'], features['body_pose'], features['jaw_pose'],
        #                         features['left_hand_pose'], features['right_hand_pose']), dim=1)
        parameters = torch.cat((features['exp'], features['body_pose'], features['jaw_pose'],
                                features['left_hand_pose'], features['right_hand_pose']), dim=1)
        return parameters


    def process_subdir(subdir, encode_pixie_dir, output_dir):
        try:
            sub_dir_path = os.path.join(encode_pixie_dir, subdir)
            pixie_video_folders = [os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) if
                                os.path.isdir(os.path.join(sub_dir_path, f))]
            audio_subdir = os.path.join(output_dir, "train", "final_features_id", subdir)
            output_subdir = os.path.join(output_dir, "train", "smoothed_final_features_id", subdir)
            # Check if the output subdirectory exists
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir, exist_ok=True)
                logging.info(f"Start processing subdirectory {subdir}")
                print(f"Start processing subdirectory {subdir}")
                # Copying corresponding audio files to output subdirectory
                p0_speak_file = os.path.join(audio_subdir,"p0_speak_audio_mfcc.npy")
                p1_speak_file = os.path.join(audio_subdir,"p1_speak_audio_mfcc.npy")
                shutil.copy(p0_speak_file, output_subdir)
                shutil.copy(p1_speak_file, output_subdir)
                # Saving final motion features to output subdirectory
                for pixie_video in pixie_video_folders:
                    pixie_files = os.path.join(pixie_video, '*_param.pkl')
                    pixie_files = sorted(filter(os.path.isfile, glob.glob(pixie_files)))
                    total_frames = len(pixie_files)
                    # N is the number of frames sequence saved in npy file each sequence is 64 frames
                    N = total_frames // 64
                    all_params = []
                    saveAll = True
                    for feature_file in tqdm(pixie_files):
                        frame_features = load_data(feature_file)
                        if saveAll:
                            all_params.append(prepare_param(frame_features))

                    video_name = os.path.basename(pixie_video[:-1])
                    # save frame features for all body parts
                    if saveAll:
                        all_params = [p.cpu().numpy() for p in all_params]
                        all_params = np.asarray(all_params)
                        # N = all_params.shape[0] // 64
                        all_params = all_params[:N * 64, :]
                        # all_params = all_params.reshape(N, 64, 365)
                        # wihout including global pose paramaters 
                        all_params = all_params.reshape(N, 64, 359)
                        np.save(os.path.join(output_subdir, video_name + '.npy'), all_params)

                logging.info(f"Finished processing subdirectory {subdir}")
                print(f"Finished processing subdirectory {subdir}")
            else:
                logging.info(f"Subdirectory {subdir} already exists in the output directory {output_dir}")
                print(f"Subdirectory {subdir} already exists in the output directory {output_dir}")

        except Exception as e:
            logging.exception(f"An error occurred while processing subdirectory {subdir}: {e}")
            print(f"An error occurred while processing subdirectory {subdir}: {e}")
            logging.error(traceback.format_exc())  # write the traceback to the log file
            traceback.print_exc(file=sys.stdout)  # print the traceback to the console


    def process_batch(batch, encode_pixie_dir, output_dir):
        try:
            logging.info(f"Start processing batch {batch}")
            print(f"Start processing batch {batch}")
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.starmap(process_subdir, [(subdir, encode_pixie_dir, output_dir) for subdir in batch])
            logging.info(f"Finished processing batch {batch}")
            print(f"Finished processing batch {batch}")
        except Exception as e:
            logging.exception(f"An error occurred while processing batch {batch}: {e}")
            print(f"An error occurred while processing batch {batch}: {e}")
            logging.error(traceback.format_exc())  # write the traceback to the log file
            traceback.print_exc(file=sys.stdout)  # print the traceback to the console


    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print("Usage: python saving_motion_features.py <input_dir> <output_dir>")
            sys.exit(1)

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]

        encode_pixie_dir = os.path.join(input_dir, "extracted_features/train/smoothed_encode_pixie_id")

        # Read subdirectories
        subdirs = [d for d in os.listdir(encode_pixie_dir) if os.path.isdir(os.path.join(encode_pixie_dir, d))]

        # Set the number of batches and number of processes
        batch_size = 15  # number of subdirectories to process in each batch
        # num_processes_per_batch = 2  # number of processes to use for each batch

        num_batches = len(subdirs) // batch_size + 1

        # Divide the subdirs into batches
        batches = [subdirs[i:i + batch_size] for i in range(0, len(subdirs), batch_size)]
        #batches = [['020150', '020149', '020025', '018020', '020090']]

        print(f"Number of subdirectories {len(subdirs)}, processed in {num_batches} batches, with batch size {batch_size}")

        # Create a pool of processes and map the batches to the processes
        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1} of {len(batches)}")
            process_batch(batch, encode_pixie_dir, output_dir)

        logging.info("All batches are processed")
        """, language = "python")


st.markdown("""
---

<sub>Learn more about PIXIE [here](https://github.com/yfeng95/PIXIE).</sub>
""", unsafe_allow_html=True)

# Subheader for Model Training Section
st.subheader("Model Training")

# Markdown Text Describing the Overall Process
st.markdown("""
After preprocessing the data, we trained our model using PyTorch. The main tasks included:

1. **Model Definition**: Defining the model layers architecture and different modules, including:
""")

# Bullet point 1: VQ-VAE
st.markdown("\t - **VQ-VAE (Vector Quantized Variational AutoEncoder):** This module quantizes listener motion features into discrete codes by mapping the input to the nearest entries in the codebook. By using these learned discrete latent codes, we ensure that the motion stays within the manifold of realistic movements, preventing any drift.")
with st.expander("VQ-VAE Module Code"):
    st.markdown("""
    ### **VQModelTransformer Class**

    In the `VQModelTransformer` class, we define the architecture for quantizing motion features and reconstructing them using learned codebook entries. The model consists of the following key components:

    - **Encoder**: The encoder, an instance of the `TransformerEncoder` class, processes the input motion features. It first downsamples the input sequence and extracts relevant features. Positional information is then embedded through Position and Linear Embeddings. A transformer with self-attention mechanisms captures the temporal dependencies in the motion data, enabling the model to effectively learn complex motion patterns.

    - **VectorQuantizer**: This module sits at the bottleneck of the VQ-VAE model. It quantizes the encoder's output by mapping it to the nearest vectors in a learned codebook. This step converts continuous motion features into discrete codes, preserving the key information while reducing dimensionality.

    - **Decoder**: The `TransformerDecoder` reconstructs the quantized sequence back into the original motion sequence. This component reverses the process of the encoder, ensuring that the decoded output closely resembles the original input motion.

    ---

    ### **setup_vq_transformer Function**

    The `setup_vq_transformer` function initializes and sets up the VQ-VAE model for both training and testing. Here’s a breakdown:

    - **Model Creation**: We create an instance of the `VQModelTransformer` and move it to the GPU for efficient computation. The model is parallelized to take advantage of multiple GPUs if available.

    - **Optimizer Setup**: A scheduled optimizer (`ScheduledOptim`) is configured with an AdamW optimizer. This optimizes the learning rate dynamically, starting with a warm-up phase, allowing the model to train effectively.

    - **Checkpoint Loading**: If a checkpoint is provided, the model and optimizer states are loaded from the saved file, resuming training from the last saved epoch. The learning rate may be adjusted if training resumes after a significant number of epochs.

    - **Starting from Scratch**: If no checkpoint is provided, the function starts training from scratch.

    The function returns the model, optimizer, and the starting epoch.

    ---

    ### **calc_vq_loss Function**

    The `calc_vq_loss` function defines the loss used for training the VQ-VAE model. It combines two key components:

    - **Reconstruction Loss**: This measures the difference between the predicted output and the ground truth motion features, ensuring that the model reconstructs realistic motion.

    - **Quantization Loss**: This loss is computed based on how effectively the input motion features are quantized into discrete codes. It encourages the model to generate meaningful and compact code representations that accurately capture the underlying motion.

    The total loss is a combination of the reconstruction loss and the weighted quantization loss, guiding the model to learn efficient and meaningful quantization of motion data.
    """)

    code_vqvae = """ 
    import functools
    import json
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import sys

    sys.path.append("../")
    from vqgan.vqmodules.quantizer import VectorQuantizer
    from modules.base_models import Transformer, PositionEmbedding, LinearEmbedding, AudioEmbedding
    from utils.base_model_util import get_activation
    from utils.optim import ScheduledOptim

    # Function to create and set up the VQ-VAE model for training or testing
    def setup_vq_transformer(args, config, load_path=None, test=False, version=None):
        # Create VQ-VAE model and optimizer
        generator = VQModelTransformer(config, version).cuda()
        learning_rate = config['learning_rate']
        print('Starting learning rate:', learning_rate)

        g_optimizer = ScheduledOptim(
            torch.optim.AdamW(generator.parameters(), betas=(0.9, 0.98), eps=1e-09),
            learning_rate,
            config['transformer_config']['hidden_size'],
            config['warmup_steps']
        )

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        generator = nn.DataParallel(generator)

        # Load model from previous checkpoint to resume training
        start_epoch = 0
        if load_path is not None:
            loaded_state = torch.load(load_path, map_location=lambda storage, loc: storage)
            generator.load_state_dict(loaded_state['state_dict'], strict=True)
            g_optimizer._optimizer.load_state_dict(loaded_state['optimizer']['optimizer'])
            g_optimizer.set_n_steps(loaded_state['optimizer']['n_steps'])
            start_epoch = loaded_state['epoch']
            print("Start epoch:", start_epoch)
            if start_epoch > 500:
                print('>> Changing learning rate to 0.00001')
                g_optimizer.set_init_lr(0.00001)
            print('Loading checkpoint from...', load_path)
        else:
            print('Starting from scratch...')
        return generator, g_optimizer, start_epoch

    # Function to compute the various components of the VQ loss
    def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
        
        # Expression Loss
        exp_loss = nn.L1Loss()(pred[:, :, :50], target[:, :, :50])
        
        # Body Poses Losses
        partbody_pose_loss = alpha * nn.L1Loss()(pred[:, :, 50:152], target[:, :, 50:152])
        neck_pose_loss = alpha * nn.L1Loss()(pred[:, :, 152:158], target[:, :, 152:158])
        head_pose_loss = alpha * nn.L1Loss()(pred[:, :, 158:164], target[:, :, 158:164])
        left_wrist_pose_loss = alpha * nn.L1Loss()(pred[:, :, 164:170], target[:, :, 164:170])
        right_wrist_pose_loss = alpha * nn.L1Loss()(pred[:, :, 170:176], target[:, :, 170:176])
        jaw_pose_loss = alpha * nn.L1Loss()(pred[:, :, 176:179], target[:, :, 176:179])
        left_hand_pose_loss = alpha * nn.L1Loss()(pred[:, :, 179:269], target[:, :, 179:269])
        right_hand_pose_loss = alpha * nn.L1Loss()(pred[:, :, 269:359], target[:, :, 269:359])

        # Combining all losses
        combined_loss = (exp_loss + partbody_pose_loss + neck_pose_loss + head_pose_loss +
                        left_wrist_pose_loss + right_wrist_pose_loss + jaw_pose_loss +
                        left_hand_pose_loss + right_hand_pose_loss)
        
        # Total Loss = VQ reconstruction + weighted pre-computed quantization loss
        total_loss = quant_loss.mean() * quant_loss_weight + combined_loss
        
        return total_loss

    # Transformer model for listener VQ-VAE
    class VQModelTransformer(nn.Module):
        def __init__(self, config, version):
            super().__init__()
            self.encoder = TransformerEncoder(config)
            self.decoder = TransformerDecoder(config, config['transformer_config']['in_dim'])
            self.quantize = VectorQuantizer(config['VQuantizer']['n_embed'],
                                            config['VQuantizer']['zquant_dim'],
                                            beta=0.30)

        def encode(self, x, x_a=None):
            h = self.encoder(x)  # x --> z'
            quant, emb_loss, info = self.quantize(h)  # finds nearest quantization
            return quant, emb_loss, info

        def decode(self, quant):
            dec = self.decoder(quant)  # z' --> x
            return dec

        def forward(self, x, x_a=None):
            quant, emb_loss, info = self.encode(x)
            dec = self.decode(quant)
            perplexity = info[0]
            return dec, emb_loss, perplexity

        def sample_step(self, x, x_a=None):
            quant_z, _, info = self.encode(x, x_a)
            x_sample_det = self.decode(quant_z)
            btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
            indices = info[2]
            x_sample_check = self.decode_to_img(indices, btc)
            return x_sample_det, x_sample_check

        def get_quant(self, x, x_a=None):
            quant_z, _, info = self.encode(x, x_a)
            indices = info[2]
            return quant_z, indices

        def get_distances(self, x):
            h = self.encoder(x)  # x --> z'
            d = self.quantize.get_distance(h)
            return d

        def get_quant_from_d(self, d, btc):
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            x = self.decode_to_img(min_encoding_indices, btc)
            return x

        @torch.no_grad()
        def decode_to_img(self, index, zshape):
            index = index.long()
            quant_z = self.quantize.get_codebook_entry(index.reshape(-1), shape=None)
            quant_z = torch.reshape(quant_z, zshape).permute(0, 2, 1)
            x = self.decode(quant_z)
            return x

        @torch.no_grad()
        def decode_logit(self, logits, zshape):
            if logits.dim() == 3:
                probs = F.softmax(logits, dim=-1)
                _, ix = torch.topk(probs, k=1, dim=-1)
            else:
                ix = logits
            ix = torch.reshape(ix, (-1, 1))
            x = self.decode_to_img(ix, zshape)
            return x
        # Function that samples the distribution of logits or returns the top indices
        def get_logit(self, logits, sample=False, filter_value=-float('Inf'),
                temperature=0.7, top_p=0.9, sample_idx=None):
            
            if sample_idx is None:
                if sample:
                    filtered_logits = self.nucleus_sampling(logits, filter_value, temperature, top_p)
                    shape = filtered_logits.shape
                    probs = F.softmax(filtered_logits, dim=-1)
                    sampled_ix = torch.multinomial(probs.view(-1, shape[-1]), 1).view(shape[:-1])
                    return sampled_ix, probs
                else:
                    probs = F.softmax(logits, dim=-1)
                    top_ix = torch.argmax(probs, dim=-1)
                    return top_ix, probs
            else:
                pass
        # Apply nucleus (top-p) sampling to logits
        def nucleus_sampling(self, logits, filter_value, temperature, top_p):
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
            return logits
    # Encoder class for VQ-VAE with Transformer backbone
    class TransformerEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            size = self.config['transformer_config']['in_dim']
            dim = self.config['transformer_config']['hidden_size']
            layers = [nn.Sequential(
                nn.Conv1d(size, dim, 5, stride=2, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm1d(dim))]
            for _ in range(1, config['transformer_config']['quant_factor']):
                layers += [nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim),
                    nn.MaxPool1d(2)
                )]
            self.squasher = nn.Sequential(*layers)
            self.encoder_transformer = Transformer(
                in_size=self.config['transformer_config']['hidden_size'],
                hidden_size=self.config['transformer_config']['hidden_size'],
                num_hidden_layers=self.config['transformer_config']['num_hidden_layers'],
                num_attention_heads=self.config['transformer_config']['num_attention_heads'],
                intermediate_size=self.config['transformer_config']['intermediate_size']
            )
            self.encoder_pos_embedding = PositionEmbedding(
                self.config["transformer_config"]["quant_sequence_length"],
                self.config['transformer_config']['hidden_size']
            )
            self.encoder_linear_embedding = LinearEmbedding(
                self.config['transformer_config']['hidden_size'],
                self.config['transformer_config']['hidden_size']
            )

        def forward(self, inputs):
            dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
            inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
            encoder_features = self.encoder_linear_embedding(inputs)
            encoder_features = self.encoder_pos_embedding(encoder_features)
            encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
            return encoder_features

     # Decoder class for VQ-VAE with Transformer backbone
    class TransformerDecoder(nn.Module):
        def __init__(self, config, out_dim, is_audio=False):
            super().__init__()
            self.config = config
            size = self.config['transformer_config']['hidden_size']
            dim = self.config['transformer_config']['hidden_size']
            self.expander = nn.ModuleList()
            self.expander.append(nn.Sequential(
                nn.ConvTranspose1d(size, dim, 5, stride=2, padding=2, output_padding=1, padding_mode='replicate'),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm1d(dim)))
            num_layers = config['transformer_config']['quant_factor'] + 2 if is_audio else config['transformer_config']['quant_factor']
            seq_len = config["transformer_config"]["sequence_length"] * 4 if is_audio else config["transformer_config"]["sequence_length"]
            for _ in range(1, num_layers):
                self.expander.append(nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim),
                ))
            self.decoder_transformer = Transformer(
                in_size=self.config['transformer_config']['hidden_size'],
                hidden_size=self.config['transformer_config']['hidden_size'],
                num_hidden_layers=self.config['transformer_config']['num_hidden_layers'],
                num_attention_heads=self.config['transformer_config']['num_attention_heads'],
                intermediate_size=self.config['transformer_config']['intermediate_size']
            )
            self.decoder_pos_embedding = PositionEmbedding(
                seq_len,
                self.config['transformer_config']['hidden_size']
            )
            self.decoder_linear_embedding = LinearEmbedding(
                self.config['transformer_config']['hidden_size'],
                self.config['transformer_config']['hidden_size']
            )
            self.cross_smooth_layer = nn.Conv1d(
                config['transformer_config']['hidden_size'],
                out_dim, 5, padding=2
            )

        def forward(self, inputs):
            dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
            for i, module in enumerate(self.expander):
                inputs = module(inputs.permute(0, 2, 1)).permute(0, 2, 1)
                if i > 0:
                    inputs = inputs.repeat_interleave(2, dim=1)
            decoder_features = self.decoder_linear_embedding(inputs)
            decoder_features = self.decoder_pos_embedding(decoder_features)
            decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
            pred_recon = self.cross_smooth_layer(decoder_features.permute(0, 2, 1)).permute(0, 2, 1)
            return pred_recon

        """

    st.code(code_vqvae, language="python")


# Bullet point 2: Cross-modal Transformer
st.markdown("\t - **Cross-modal Transformer:** Aligns the speaker motion and audio inputs into a shared latent space using a cross-attention mechanism")
with st.expander("Cross-modal Transformer Module Code"):
    st.markdown("""
    The **Cross-modal Transformer** using a cross-attention mechanism to effectively integrate multi-modal inputs. It processes audio inputs as queries and motion data as keys and values. This architecture is implemented using multiple layers, each composed of `CrossModalAttention` , `Residual` , `Norm`, and `MLP` modules. The `CrossModalAttention` layer performs cross-attention between modalities, while the `MLP` layers further refine the fused features. The goal is to capture long-range dependencies between the speaker's motion and audio inputs, aligning them in a shared latent space. The model can switch between cross-modal and self-attention modes depending on the configuration of the 'Transformer' class.
    """)
    code_transformer = """
    class Transformer(nn.Module):
        def __init__(self, in_size=50, hidden_size=768, num_hidden_layers=12, 
                     num_attention_heads=12, intermediate_size=3072, 
                     cross_modal=False, in_dim2=None):
            super().__init__()
            blocks = []
            if cross_modal:
                for i in range(num_hidden_layers):
                    blocks.extend([
                        Residual(Norm(CrossModalAttention(in_size, hidden_size, 
                                                         heads=num_attention_heads, 
                                                         in_dim2=in_dim2), hidden_size)),
                        Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size), hidden_size))
                    ])
            else:
                for i in range(num_hidden_layers):
                    blocks.extend([
                        Residual(Norm(Attention(in_size, hidden_size, 
                                                heads=num_attention_heads), hidden_size)),
                        Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size), hidden_size))
                    ])
            self.net = nn.Sequential(*blocks)

        def forward(self, x_data):
            if self.cross_modal:
                x_data = self.net(x_data)
                return x_data['x_b']
            else:
                x, mask_info = x_data
                x, _ = self.net((x, mask_info))
                return x
    """
    st.code(code_transformer, language="python")

# Bullet point 3: Personlity Network
st.markdown("\t - **Personality Network:** This module conditions listener sequences on personality traits by embedding personality scores, such as extraversion, into a higher-dimensional vector space. By using a Multi-Layer Perceptron (MLP) network, the personality values are transformed into a conditioned embedding.")

with st.expander("Personality Network Module Code"):
    st.markdown("""
    ### **ConditioningMLP Class**

    The `ConditioningMLP` class defines a simple MLP architecture for embedding personality scores. The model takes as input a sequence of personality scores and transforms them using the following components:

    - **Input Layer**: Maps the personality score to a higher-dimensional space.
    - **Hidden Layer**: A non-linear transformation using GELU activation to capture complex relationships.
    - **Output Layer**: Produces the final conditioned vector that represents the personality in a higher-dimensional space.

    The input sequence is reshaped before passing through the network and is then reshaped back into the original sequence format with the newly transformed embeddings.
    """)
    code_personality_network = """
    import torch.nn as nn
    import torch.nn.functional as F

    class ConditioningMLP(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, output_size=128):
            super(ConditioningMLP, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

            # Define layers of the MLP
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        def forward(self, x):
            # x shape: [B, seq_len, 1]
            B, seq_len, _ = x.shape
            x = x.view(-1, self.input_size)  # Flatten input for MLP
            
            # Apply the MLP with GELU activation
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            
            # Reshape back to [B, seq_len, output_size]
            x = x.view(B, seq_len, -1)
            return x
        """
    st.code(code_personality_network, language="python")
# Bullet point 4: Predictor Transformer
st.markdown("\t - **Predictor Transformer:** Our full attention transformer captures the correlation between the speaker motion and audio sequences and the listener's past motion sequences to autoregressively predict the future listener motion sequence.")

# Expander for Predictor Transformer Module Code
with st.expander("Predictor Transformer Module Code"):
    st.markdown("""
    The **Predictor Transformer** utilizes two modalities: 
    - **modal_a_sequences** represents the listener's motion quantized sequence (first modality), potentially concatenated with personality network embeddings. 
    - **modal_b_sequences** represents the second modality, which includes fused outputs from a previous cross-modal transformer handling speaker motion and audio embedding.

    The inputs are concatenated and passed through position embeddings and a Transformer model with multiple layers of attention. Additionally, binary masks are applied to regulate the attention mechanism. After the Transformer, the output is normalized using `LayerNorm` and passed through a final linear output layer to produce the `logits` representing the predicted listener motion.
    """)
    
    code_predictor_transformer = """
    class CrossModalLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            model_config = self.config['transformer']
            self.transformer_layer = Transformer(
                in_size=model_config['hidden_size'],
                hidden_size=model_config['hidden_size'],
                num_hidden_layers=model_config['num_hidden_layers'],
                num_attention_heads=model_config['num_attention_heads'],
                intermediate_size=model_config['intermediate_size'])

            output_layer_config = self.config['output_layer']
            self.cross_norm_layer = nn.LayerNorm(self.config['in_dim'])
            self.cross_output_layer = nn.Linear(
                                        self.config['in_dim'],
                                        output_layer_config['out_dim'],
                                        bias=False)

            self.cross_pos_embedding = PositionEmbedding(
                    self.config["sequence_length"], self.config['in_dim'])

        def forward(self, modal_a_sequences, modal_b_sequences, mask_info):
            _, _, modal_a_width = get_shape_list(modal_a_sequences)
            merged_sequences = modal_a_sequences
            if modal_b_sequences is not None:
                _, _, modal_b_width = get_shape_list(modal_b_sequences)
                if modal_a_width != modal_b_width:
                    raise ValueError(
                        "The modal_a hidden size (%d) should be the same with the modal_b"
                        "hidden size (%d)" % (modal_a_width, modal_b_width))
                merged_sequences = torch.cat([merged_sequences, modal_b_sequences], axis=1)

            merged_sequences = self.cross_pos_embedding(merged_sequences)
            merged_sequences = self.transformer_layer((merged_sequences, mask_info))
            merged_sequences = self.cross_norm_layer(merged_sequences)
            logits = self.cross_output_layer(merged_sequences)
            return logits
    """
    
    # Display code in the expander
    st.code(code_predictor_transformer, language="python")

# Continue the markdown for the remaining points
st.markdown("""
2. **Training Loop:** Implementing the training loop, including loss computation and backpropagation. Our model training consists of two stages:
""")

st.markdown("""
\t - **Stage 1: Training the VQ-VAE:** We start by training the VQ-VAE model, which is responsible for learning the listener codebook entries. 
    The VQ-VAE is optimized using the AdamW optimizer, which is configured with a scheduled learning rate decay. 
    Specifically, we use an initial learning rate of 0.1 and a warm-up phase of 100,000 steps. The batch size for training is set to 32, and we train the model for around 1300 epochs, which took approximately 3 days using 8 GPUs. 
    The training dataset consists of listener sequences of length 32, and the encoder, quantizer, and decoder components of the VQ-VAE are trained jointly to minimize the reconstruction loss as well as the quantization loss.
""")
with st.expander("VQ-VAE Training Code"):
    st.markdown("""
    ### *Training the VQ-VAE*

    The VQ-VAE model is trained using the following code. The training loop iterates over batches of listener motion sequences, computing the reconstruction loss and quantization loss. The optimizer performs backpropagation to update the model's weights.

     - **Experiment Logging and Management: ** For each training session, we create a unique experiment directory with a timestamp, saving metadata, model checkpoints, and training logs. Configuration files are preserved for reproducibility, and training progress is logged with TensorBoard for easy monitoring of metrics such as loss.
    
    - **`generator_train_step`: ** This function handles the autoencoding training for the VQ-VAE model. It processes the input listener sequences in batches, computes the reconstruction and quantization losses, and performs backpropagation to update the model weights. Throughout each epoch, the function tracks the total training loss and logs it using TensorBoard. The optimizer's learning rate is updated at each step.    

    - **`generator_val_step`: ** This function evaluates the VQ-VAE model on the validation set after each epoch. It calculates the average validation loss and logs it using TensorBoard. If the current validation loss is lower than the defined threshold, the model's state is saved as the best checkpoint. Additionally, this function includes an early stopping mechanism where training halts if no significant improvement is observed, ensuring efficient training..

    """)

    code_training_quantizer = """
    import argparse
    import json
    import subprocess
    import logging
    import numpy as np
    import os
    import pdb
    import scipy.io as sio

    import torch
    from torch import nn
    from torch.autograd import Variable
    import torchvision
    from torch.utils.tensorboard import SummaryWriter

    from vqmodules.gan_models import setup_vq_transformer, calc_vq_loss
    import sys
    import shutil

    sys.path.append('../')
    from utils.load_utils import *
    from datetime import datetime


    def generator_train_step(config, epoch, generator, g_optimizer, train_X,
                            rng, writer):
        # Function to do autoencoding training for VQ-VAE

    generator.train()
    batchinds = np.arange(train_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)
    total_epoch_loss = 0.0  # To accumulate loss over the entire epoch
    for bii, bi in enumerate(batchinds):
        idxStart = bi * config['batch_size']
        gtData_np = train_X[idxStart:(idxStart + config['batch_size']), :, :]
        gtData = Variable(torch.from_numpy(gtData_np),
                          requires_grad=False).cuda()
        prediction, quant_loss, perplexity = generator(gtData, None)
        g_loss = calc_vq_loss(prediction, gtData, quant_loss)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step_and_update_lr()
        total_epoch_loss += g_loss.detach().item()
        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}' \
                  .format(epoch, config['num_epochs'], bii, totalSteps,
                          g_loss.detach().item(), perplexity[0].item()))

    avg_epoch_loss = total_epoch_loss / totalSteps
    writer.add_scalar('Loss/train_totalLoss', avg_epoch_loss, epoch)



    def generator_val_step(config, epoch, generator, g_optimizer, test_X,
                        currBestLoss, prev_save_epoch, tag, writer):
        # Function that validates training of VQ-VAE

        generator.eval()
        batchinds = np.arange(test_X.shape[0] // config['batch_size'])
        totalSteps = len(batchinds)
        total_val_loss = 0.0  # To accumulate loss over the entire validation epoch
        for bii, bi in enumerate(batchinds):
            idxStart = bi * config['batch_size']
            gtData_np = test_X[idxStart:(idxStart + config['batch_size']), :, :]
            gtData = Variable(torch.from_numpy(gtData_np),
                            requires_grad=False).cuda()
            with torch.no_grad():
                prediction, quant_loss, perplexity = generator(gtData, None)
            g_loss = calc_vq_loss(prediction, gtData, quant_loss)
            total_val_loss += g_loss.detach().item()

        avg_val_loss = total_val_loss / totalSteps
        print('val_Epoch [{}/{}], Average Loss: {:.4f}, Perplexity: {:5.4f}' \
            .format(epoch, config['num_epochs'], avg_val_loss, perplexity[0].item()))
        print('----------------------------------')
        writer.add_scalar('Loss/val_totalLoss', avg_val_loss, epoch)

        ## save model if curr loss is less than 25
        if avg_val_loss < 25:
            prev_save_epoch = epoch
            checkpoint = {'config': args.config,
                        'state_dict': generator.state_dict(),
                        'optimizer': {
                            'optimizer': g_optimizer._optimizer.state_dict(),
                            'n_steps': g_optimizer.n_steps,
                        },
                        'epoch': epoch}
            fileName = config['model_path'] + \
                    '{}{}_best.pth'.format(tag, config['pipeline'])
            print('>>>> saving best epoch {}'.format(epoch), avg_val_loss)
            torch.save(checkpoint, fileName)
            # Add graph to tensorboard

        return currBestLoss, prev_save_epoch, avg_val_loss

    def main(args):
        
        rng = np.random.RandomState(23456)
        torch.manual_seed(23456)
        torch.cuda.manual_seed(23456)
        print('using config', args.config)
        with open(args.config) as f:
            config = json.load(f)
        tag = config['tag']
        pipeline = config['pipeline']
        # create a timestamp string to use in the folder name
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # create a new folder for this experiment
        experiment_dir = os.path.join('experiments/round_3', f'experiment_{timestamp}')
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        # Prompt the user for experiment details
        experiment_type = input("What type of features do you want to quantize (speaking/listening)? ")
        remarks = input("Any remarks or description about the experiment? ")

        # Create a README for the experiment
        with open(os.path.join(experiment_dir, "README.md"), "w") as f:
            f.write(f"# Experiment Details: {timestamp}\n")
            f.write(f"\n## Experiment Type: {experiment_type}\n")
            f.write("\n## Hyperparameters:\n")
            for key, value in config.items():
                f.write(f"- {key}: {value}\n")
            f.write(f"\n## Remarks:\n{remarks}\n")

        # set the models and runs directories to be subdirectories of the experiment directory
        models_dir = os.path.join(experiment_dir, 'models/')
        runs_dir = os.path.join(experiment_dir, 'runs/')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)
        # save the models_dir path in the configuration file
        # config['model_path'] = models_dir
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        currBestLoss = 1e3
        ## can modify via configs, these are default for released model
        seq_len = 32
        prev_save_epoch = 0
        with open(args.config) as f:
            config = json.load(f)

        # Save the updated configuration file in the experiment directory
        config_file = os.path.basename(args.config)
        experiment_config_path = os.path.join(experiment_dir, config_file)
        shutil.copyfile(args.config, experiment_config_path)
        # pdb.set_trace()

        writer = SummaryWriter(os.path.join(runs_dir, 'debug_{}{}').format(tag, pipeline))
        ## setting up models and loading last runned checkpoint
        fileName = config['model_path'] + \
                '{}{}_best.pth'.format(tag, config['pipeline'])
        load_path = fileName if os.path.exists(fileName) else None
        generator, g_optimizer, start_epoch = setup_vq_transformer(args, config,
                                                                version=None, load_path=load_path)
        generator.train()

        config['model_path'] = models_dir

        ## training/validation process
        _, _, train_listener, test_listener, _, _ , _ , _ ,= \
            load_data(config, pipeline, tag, rng,
                    segment_tag=config['segment_tag'], smooth=False)
        train_X = np.concatenate((train_listener[:, :seq_len, :],
                                train_listener[:, seq_len:, :]), axis=0)
        test_X = np.concatenate((test_listener[:, :seq_len, :],
                                test_listener[:, seq_len:, :]), axis=0)
        print('loaded listener...', train_X.shape, test_X.shape)
        disc_factor = 0.0
        for epoch in range(start_epoch, start_epoch + config['num_epochs']):
            print('epoch', epoch, 'num_epochs', config['num_epochs'])
            if epoch == start_epoch + config['num_epochs'] - 1:
                print('early stopping at:', epoch)
                print('best loss:', currBestLoss)
                break
            generator_train_step(config, epoch, generator, g_optimizer, train_X,
                                rng, writer)
            currBestLoss, prev_save_epoch, g_loss = \
                generator_val_step(config, epoch, generator, g_optimizer, test_X,
                                currBestLoss, prev_save_epoch, tag, writer)
        print('final best loss:', currBestLoss)


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--checkpoint', type=str, default=None)
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--ar_load', action='store_true')
        args = parser.parse_args()
        main(args)
        """
    
    # Display code in the expander
    st.code(code_training_quantizer, language="python")

# Load the validation loss image image
model_vq_loss = Image.open('Images/validation-loss.png')

# Convert the image to base64
buffer = BytesIO()
model_vq_loss.save(buffer, format='PNG')
img_str = base64.b64encode(buffer.getvalue()).decode()

image_html = f"""
    <div style="text-align: center;">
        <img style="height: 350px; width: auto;" src="data:image/png;base64,{img_str}" alt="Model Architecture">
    </div>
"""

# Display the image
st.markdown(image_html, unsafe_allow_html=True)


st.markdown("""
\t - **Stage 2: Training the full model:** After training the VQ-VAE and learning the codebook, we discretize the listener's past motion sequences and match them to the closest codebook entries. These are concatenated with personality embeddings and combined with speaker motion and audio features using a cross-modal transformer.

\t \t We then train the full model, including the predictor transformer, to predict future listener motion. We trained the model (including the predictor and cross-modal transformer) for 1,200 epochs, which took nearly 1.5 days on 8 GPUs. We used a learning rate of 0.001 with 50,000 warm-up steps and a batch size of 32.
""")


with st.expander("Training Full Model Code"):
    st.markdown("""
    ### *Training the model*

     The training pipeline for the model involves several key steps:
    
    1. **Data Preparation**: 
        - **`gather_data` Function**: This function prepares the data by discretizing past listener motion sequences using a pre-trained VQ-VAE codebook. It formats the data into tensors suitable for training, including speaker and listener motion data, audio data, and personality metadata.

    2. **Training Step**:
        - **`generator_train_step` Function**: This function trains the Predictor model, which forecasts future listener motion based on past listener motion and speaker data. It computes the loss, performs backpropagation, and updates model parameters. The function logs the loss and perplexity metrics to TensorBoard for monitoring.

    3. **Validation Step**:
        - **`generator_val_step` Function**: This function evaluates the performance of the Predictor model on the validation dataset. It calculates the average validation loss and saves the model checkpoint if the performance improves.

    4. **Main Function**:
        - **`main` Function**: This orchestrates the entire training process. It initializes the models and optimizers, loads data, and iterates through epochs to train and validate the model. It also manages experiment directories and configuration.

    
    """)
    code_training_full_model = """
    import argparse
    import json
    import logging
    import numpy as np
    import os
    import shutil
    import torch
    from torch import nn
    from torch.autograd import Variable
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    from modules.fact_model import setup_model, calc_logit_loss
    from vqgan.vqmodules.gan_models import setup_vq_transformer
    from utils.base_model_util import *
    from utils.load_utils import *

    def gather_data(config, X, Y, audio, meta, l_vq_model, patch_size, seq_len, bi):
        idxStart = bi * config['batch_size']
        speakerData_np = X[idxStart:(idxStart + config['batch_size']), :, :]
        listenerData_np = Y[idxStart:(idxStart + config['batch_size']), :, :]
        audioData_np = audio[idxStart:(idxStart + config['batch_size']), :, :]
        personalityData_np = meta[idxStart:(idxStart + config['batch_size']), :]
        inputs, listener_future, raw_listener, btc = \
            create_data_vq(l_vq_model, speakerData_np, listenerData_np,
                           audioData_np, personalityData_np, seq_len,
                           data_type=config['loss_config']['loss_type'],
                           patch_size=patch_size)
        return inputs, listener_future, raw_listener, btc

    def generator_train_step(config, epoch, generator, g_optimizer, l_vq_model,
                             train_X, train_Y, train_audio, train_meta, rng, writer,
                             patch_size, seq_len):
        generator.train()
        batchinds = np.arange(train_X.shape[0] // config['batch_size'])
        totalSteps = len(batchinds)
        rng.shuffle(batchinds)
        total_epoch_loss = 0.0
        for bii, bi in enumerate(batchinds):
            inputs, listener_future, _, _ = gather_data(config, train_X, train_Y,
                                                        train_audio, train_meta, l_vq_model,
                                                        patch_size, seq_len, bi)
            prediction = generator(inputs,
                                   config['fact_model']['cross_modal_model']['max_mask_len'],
                                   -1)
            cut_point = listener_future.shape[1]
            logit_loss = calc_logit_loss(prediction[:, :cut_point, :800],
                                         listener_future[:, :cut_point])
            g_loss = logit_loss
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step_and_update_lr()
            total_epoch_loss += g_loss.detach().item()
            if bii % config['log_step'] == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}' \
                      .format(epoch, config['num_epochs'], bii, totalSteps,
                              g_loss.detach().item(), np.exp(total_epoch_loss / totalSteps)))
        avg_epoch_loss = total_epoch_loss / totalSteps
        writer.add_scalar('Loss/train_totalLoss', avg_epoch_loss, epoch)

    def generator_val_step(config, epoch, generator, g_optimizer, l_vq_model,
                           test_X, test_Y, test_audio, test_meta, currBestLoss,
                           prev_save_epoch, tag, writer, patch_size, seq_len):
        generator.eval()
        batchinds = np.arange(test_X.shape[0] // config['batch_size'])
        totalSteps = len(batchinds)
        total_val_loss = 0.0
        for bii, bi in enumerate(batchinds):
            inputs, listener_future, _, _ = gather_data(config, test_X, test_Y,
                                                        test_audio, test_meta, l_vq_model,
                                                        patch_size, seq_len, bi)
            with torch.no_grad():
                prediction = generator(inputs,
                                       config['fact_model']['cross_modal_model']['max_mask_len'], -1)
            cut_point = listener_future.shape[1]
            logit_loss = calc_logit_loss(prediction[:, :cut_point, :800],
                                         listener_future[:, :cut_point])
            g_loss = logit_loss
            total_val_loss += g_loss.detach().item()
        avg_val_loss = total_val_loss / totalSteps
        print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}' \
              .format(epoch, config['num_epochs'], bii, totalSteps,
                      avg_val_loss, np.exp(avg_val_loss)))
        if avg_val_loss < currBestLoss:
            prev_save_epoch = epoch
            checkpoint = {'config': args.config,
                          'state_dict': generator.state_dict(),
                          'optimizer': {
                              'optimizer': g_optimizer._optimizer.state_dict(),
                              'n_steps': g_optimizer.n_steps,
                          },
                          'epoch': epoch}
            fileName = config['model_path'] + \
                       '{}{}_best.pth'.format(tag, config['pipeline'])
            currBestLoss = avg_val_loss
            torch.save(checkpoint, fileName)
            print('>>>> saving best epoch {}'.format(epoch), avg_val_loss)
        return currBestLoss, prev_save_epoch, avg_val_loss

    def main(args):
        rng = np.random.RandomState(23456)
        torch.manual_seed(23456)
        torch.cuda.manual_seed(23456)
        print('using config', args.config)
        with open(args.config) as f:
            config = json.load(f)
        tag = config['tag']
        pipeline = config['pipeline']
        codebook_experiment = config['l_vqconfig']
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_dir = os.path.join('experiments/round_4', f'experiment_{timestamp}')
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        remarks = input("Any remarks or description about the experiment? ")
        with open(os.path.join(experiment_dir, "README.md"), "w") as f:
            f.write(f"# Experiment Details: {timestamp}\n")
            f.write(f"\n## Experiment Type: {experiment_type}\n")
            f.write(f"\n## Code-Book: {codebook_experiment}\n")
            f.write("\n## Hyperparameters:\n")
            for key, value in config.items():
                f.write(f"- {key}: {value}\n")
            f.write(f"\n## Remarks:\n{remarks}\n")
        models_dir = os.path.join(experiment_dir, 'models/')
        runs_dir = os.path.join(experiment_dir, 'runs/')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)
        config['model_path'] = models_dir
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        with open(args.config) as f:
            config = json.load(f)
        config_file = os.path.basename(args.config)
        experiment_config_path = os.path.join(experiment_dir, config_file)
        shutil.copyfile(args.config, experiment_config_path)
        writer = SummaryWriter(os.path.join(runs_dir, 'debug_{}{}').format(tag, pipeline))
        args.get_attn = False
        currBestLoss = 1e3
        prev_save_epoch = 0
        patch_size = 8
        seq_len = 32
        with open(config['l_vqconfig']) as f:
            l_vqconfig = json.load(f)
        l_model_path = 'vqgan/' + l_vqconfig['model_path'] + \
                       '{}{}_best.pth'.format(l_vqconfig['tag'], l_vqconfig['pipeline'])
        l_vq_model, _, _ = setup_vq_transformer(args, l_vqconfig,
                                                load_path=l_model_path)
        for param in l_vq_model.parameters():
            param.requires_grad = False
        l_vq_model.eval()
        vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}
        fileName = config['model_path'] + \
                   '{}{}_best.pth'.format(tag, config['pipeline'])
        load_path = fileName if os.path.exists(fileName) else None
    """
    st.code(code_training_full_model, language="python")

st.markdown("""
3. **Evaluation:** Evaluating the final model performance on the test set.
""")

with st.expander("Evaluation Code"):
    st.markdown("""
    ### *Evaluation Metrics*

    The script calculates several key metrics for evaluating final model performance. Each metric is explained below:

    1. **L2 Distance Calculation**:
       - **`l2_distance` Function**: Computes the L2 (Euclidean) distance between the reference and predicted data for both facial and body features. This provides a measure of the overall deviation between the generated and ground truth.

    2. **Frechet Feature Distance (FD)**:
       - **`calculate_frechet_feature_distance` Function**: Computes the Frechet Distance between the reference and predicted feature distributions for both face and body. This assesses the similarity of feature distributions by comparing their means and covariances, offering a more detailed comparison than L2 distance.

    3. **Paired Frechet Feature Distance (P-FD)**:
       - **`paired_frechet_feature_distance` Function**: Extends the Frechet Distance by incorporating speaker features along with listener data, comparing concatenated features of both. This helps evaluate the interaction dynamics between speaker and listener.

    4. **Temporal Diversity**:
       - **`calculate_temporal_diversity` Function**: Measures the temporal variance of predicted sequences, offering insight into the model ability to generate diverse and dynamic behaviors over time.
    """)
    code_evaluation = """
    import argparse
    import json
    import numpy as np
    from scipy import linalg
    from utils.load_utils import *
    # import pdb

    def l2_distance(ref, pred):
        # extracting facial features (exp and jaw_pose)
        ref_face = np.concatenate((ref[:, :, :50], ref[:, :, 176:179]), axis=2)
        pred_face = np.concatenate((pred[:, :, :50], pred[:, :, 176:179]), axis=2)
        
        # extract body features (excluding exp and jaw_pose)
        ref_body = np.concatenate((ref[:, :, 50:176], ref[:, :, 179:]), axis=2)
        pred_body = np.concatenate((pred[:, :, 50:176], pred[:, :, 179:]), axis=2)
            
        # calculate L2 distance for face features
        face_l2 = np.mean(np.linalg.norm(pred_face - ref_face, axis=-1))
        
        # calculate L2 distance for body features
        body_l2 = np.mean(np.linalg.norm(pred_body - ref_body, axis=-1))
        
        return face_l2, body_l2

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

        Code apapted from https://github.com/mseitzer/pytorch-fid

        Copyright 2018 Institute of Bioinformatics, JKU Linz
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.

        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        mu and sigma are calculated through:
        ```
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        ```
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
    
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def calculate_frechet_feature_distance(ref, pred):
        # extracting facial features (exp and jaw_pose)
        ref_face = np.concatenate((ref[:, :, :50], ref[:, :, 176:179]), axis=2)
        pred_face = np.concatenate((pred[:, :, :50], pred[:, :, 176:179]), axis=2)
        
        # extract body features (excluding exp and jaw_pose)
        ref_body = np.concatenate((ref[:, :, 50:176], ref[:, :, 179:]), axis=2)
        pred_body = np.concatenate((pred[:, :, 50:176], pred[:, :, 179:]), axis=2)
        
        # reshape features
        ref_face = ref_face.reshape(-1, ref_face.shape[2])
        pred_face = pred_face.reshape(-1, pred_face.shape[2])
        ref_body = ref_body.reshape(-1, ref_body.shape[2])
        pred_body = pred_body.reshape(-1, pred_body.shape[2])

        # calculate mean and covariance for face features
        mean_ref_face = np.mean(ref_face, axis=0)
        mean_pred_face = np.mean(pred_face, axis=0)
        cov_ref_face = np.cov(ref_face, rowvar=False)
        cov_pred_face = np.cov(pred_face, rowvar=False)

        # calculate mean and covariance for body features
        mean_ref_body = np.mean(ref_body, axis=0)
        mean_pred_body = np.mean(pred_body, axis=0)
        cov_ref_body = np.cov(ref_body, rowvar=False)
        cov_pred_body = np.cov(pred_body, rowvar=False)

        # normalization for face features
        std_face = np.std(ref_face, axis=0) + 1e-10
        ref_face = (ref_face - mean_ref_face) / std_face
        pred_face = (pred_face - mean_ref_face) / std_face

        # normalization for body features
        std_body = np.std(ref_body, axis=0) + 1e-10
        ref_body = (ref_body - mean_ref_body) / std_body
        pred_body = (pred_body - mean_ref_body) / std_body

        # calculate Frechet distance for face features
        fdist_face = calculate_frechet_distance(
            mu1=np.mean(ref_face, axis=0), 
            sigma1=np.cov(ref_face, rowvar=False),
            mu2=np.mean(pred_face, axis=0), 
            sigma2=np.cov(pred_face, rowvar=False),
        )

        # calculate Frechet distance for body features
        fdist_body = calculate_frechet_distance(
            mu1=np.mean(ref_body, axis=0), 
            sigma1=np.cov(ref_body, rowvar=False),
            mu2=np.mean(pred_body, axis=0), 
            sigma2=np.cov(pred_body, rowvar=False),
        )
        
        return fdist_face, fdist_body

    def paired_frechet_feature_distance(ref, pred, speak):
        # extracting concatenated facial features (exp and jaw_pose)
        ref_face = np.concatenate((ref[:, :, :50], ref[:, :, 176:179], speak[:, :, :50], speak[:, :, 176:179]), axis=-1)
        pred_face = np.concatenate((pred[:, :, :50], pred[:, :, 176:179], speak[:, :, :50], speak[:, :, 176:179]), axis=-1)
        
        # extract concatenated body features (excluding exp and jaw_pose)
        ref_body = np.concatenate((ref[:, :, 50:176], ref[:, :, 179:], speak[:, :, 50:176], speak[:, :, 179:]), axis=-1)
        pred_body = np.concatenate((pred[:, :, 50:176], pred[:, :, 179:], speak[:, :, 50:176], speak[:, :, 179:]), axis=-1)
        
        # reshape features
        ref_face = ref_face.reshape(-1, ref_face.shape[2])
        pred_face = pred_face.reshape(-1, pred_face.shape[2])
        ref_body = ref_body.reshape(-1, ref_body.shape[2])
        pred_body = pred_body.reshape(-1, pred_body.shape[2])

        # calculate mean and covariance for face features
        mean_ref_face = np.mean(ref_face, axis=0)
        mean_pred_face = np.mean(pred_face, axis=0)
        cov_ref_face = np.cov(ref_face, rowvar=False)
        cov_pred_face = np.cov(pred_face, rowvar=False)

        # calculate mean and covariance for body features
        mean_ref_body = np.mean(ref_body, axis=0)
        mean_pred_body = np.mean(pred_body, axis=0)
        cov_ref_body = np.cov(ref_body, rowvar=False)
        cov_pred_body = np.cov(pred_body, rowvar=False)

        # normalization for face features
        std_face = np.std(ref_face, axis=0) + 1e-10
        ref_face = (ref_face - mean_ref_face) / std_face
        pred_face = (pred_face - mean_ref_face) / std_face

        # normalization for body features
        std_body = np.std(ref_body, axis=0) + 1e-10
        ref_body = (ref_body - mean_ref_body) / std_body
        pred_body = (pred_body - mean_ref_body) / std_body

        # calculate Frechet distance for face features
        fdist_face = calculate_frechet_distance(
            mu1=np.mean(ref_face, axis=0), 
            sigma1=np.cov(ref_face, rowvar=False),
            mu2=np.mean(pred_face, axis=0), 
            sigma2=np.cov(pred_face, rowvar=False),
        )

        # calculate Frechet distance for body features
        fdist_body = calculate_frechet_distance(
            mu1=np.mean(ref_body, axis=0), 
            sigma1=np.cov(ref_body, rowvar=False),
            mu2=np.mean(pred_body, axis=0), 
            sigma2=np.cov(pred_body, rowvar=False),
        )
        
        return fdist_face, fdist_body

    def calculate_temporal_diversity(predictions):
        # Calculate temporal variance for face_features (exp and jaw_pose)
        exp_features = predictions[:, :, :50]  # Extracting exp features (first 50 features)
        jaw_pose_features = predictions[:, :, 176:179]  # Extracting jaw_pose features (indices 176 to 178)
        face_features = np.concatenate((exp_features, jaw_pose_features), axis=2)
        temporal_variance_face = np.var(face_features, axis=1)


        # Calculate temporal variance for body_features (excluding exp and jaw_pose)
        body_features_part1 = predictions[:, :, 50:176]  # Features between exp and jaw_pose
        body_features_part2 = predictions[:, :, 179:]  # Features after jaw_pose
        body_features = np.concatenate((body_features_part1, body_features_part2), axis=2)
        temporal_variance_body = np.var(body_features, axis=1)


        # Average the variance over both batches (B) and features (F)
        diversity_score_face = np.mean(temporal_variance_face)
        diversity_score_body = np.mean(temporal_variance_body)

        return diversity_score_face, diversity_score_body


        def main(args):


        with open(args.config) as f:
            config = json.load(f)
        pipeline = config['pipeline']
        tag = config['tag']

        ## setup VQ-VAE model
        with open(config['l_vqconfig']) as f:
            l_vqconfig = json.load(f)

        vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}

        ## load reference data or pixie Input predictions
        out_num = 1
        test_X, test_Y, test_audio, test_meta, _ = \
            load_test_data(config, pipeline, tag, out_num=out_num,
                        vqconfigs=vq_configs, smooth=True,
                        speaker=args.speaker,eval_mode = 'u', num_out=None)

        B, T, F = test_Y.shape[0], test_Y.shape[1], test_Y.shape[2]
        test_Y = test_Y.reshape(B * 2, T // 2, F)
        test_X = test_X.reshape(B * 2, T // 2, F)
        test_Y = test_Y[1:, : , :]
        test_X = test_X[:, : , :]
        
        pred = np.load('outputs/all_udiva/conditioned/conditioned_model_actual_pred_all_speaker.npy')
        ref = test_Y[:pred.shape[0], :, :]
        speak = test_X[:pred.shape[0], :, :]

        # pdb.set_trace()

        # L2 Distance
        face_l2, body_l2 = l2_distance(ref, pred)
        print("L2 Distance - face: ", round(face_l2, 3))
        print("L2 Distance - Body: ", round(body_l2, 3))

        # Frechet Feature Distance
        fdist_face, fdist_body = calculate_frechet_feature_distance(ref, pred)
        print("Frechet Distance (FD) - face: ", round(fdist_face, 3))
        print("Frechet Distance (FD)- Body: ", round(fdist_body, 3))

        # Paired Frechet Feature Distance with Speaker Features
        pfdist_face, pfdist_body = paired_frechet_feature_distance(ref, pred, speak)
        print("Paired Frechet Distance (P-FD) with Listener & Speaker - face: ", round(pfdist_face, 3))
        print("Paired Frechet Distance (P-FD) with Listener & Speaker - Body: ", round(pfdist_body, 3))

        # Temporal Diversity
        diversity_score_face, diversity_score_body = calculate_temporal_diversity(pred)
        print("Temporal Diversity Score (Variance) - face:", round(diversity_score_face, 3))
        print("Temporal Diversity Score (Variance)- Body:", round(diversity_score_body, 3))

        # Run the main function
        if __name__ == '__main__':
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', type=str, required = True)
            parser.add_argument('--speaker', type=str, required = True)
            args = parser.parse_args()
            print(args)
            main(args)
    """
    st.code(code_evaluation, language="python")


# Applications
st.markdown("<a name='applications'></a>", unsafe_allow_html=True)
st.header("Applications")
st.markdown("""The ability to accurately model dyadic conversations and generate listener avatars with personality-driven non-verbal cues has numerous practical applications. In this section, we explore three potential applications of our research in the fields of personalized virtual assistants, virtual customer service agents, and virtual therapy and counseling.

**Personalized virtual assistants** have become increasingly popular in recent years, as they offer a convenient, and intuitive way for users to interact with technology. However, most virtual assistants still rely primarily on verbal cues and lack the ability to exhibit realistic, and engaging non-verbal behaviors. By incorporating personality traits into our model, we can create virtual assistants that adapt their behavior to match the user's personality or mood, enhancing user engagement and satisfaction.

**Virtual customer service agents** are another promising application of our research. In today's competitive business environment, providing a personalized and engaging customer experience is essential for success. Our model can be used to create virtual agents that exhibit personality-driven non-verbal cues, allowing them to tailor their behavior to match the customer's personality and create a more effective and satisfying interaction.

**virtual therapy and counseling** are emerging fields that could greatly benefit from our research. By creating virtual therapists or counselors that exhibit empathetic and supportive non-verbal behaviors, we can make virtual therapy sessions more effective and comforting for users. Our model can be used to simulate different personality types, allowing users to choose a virtual therapist or counselor that matches their own personality and preferences.""")


# Load the image
model_architecture = Image.open('Images/conversational-agent.jpg')

# Convert the image to base64
buffer = BytesIO()
model_architecture.save(buffer, format='PNG')
img_str = base64.b64encode(buffer.getvalue()).decode()

image_html = f"""
    <div style="text-align: center;">
        <img style="height: 600px; width: auto;" src="data:image/png;base64,{img_str}" alt="Model Architecture">
    </div>
"""

# Display the image
st.markdown(image_html, unsafe_allow_html=True)


