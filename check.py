# import streamlit as st
# import torch
# from google import genai
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
# model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
# labels = ['anger',
# 'disgust',
# 'fear',
# 'joy',
# 'neutral',
# 'sadness',
# 'surprise']

# def get_emotion(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#     predicted_class = torch.argmax(probs, dim=1).item()
#     return labels[predicted_class]

# print(get_emotion("I feel so happy and excited!"))  
# def generate_poem(mood, language):
#     client = genai.Client(api_key="AIzaSyCxAu00RguxS3mo-b025nF4FtIg1fVvlkY")
#     prompt = f"Generate a {mood} poem  in {language}."
#     response = client.models.generate_content(
#         model="gemini-2.0-flash", contents=prompt
#     )
#     return response.text if hasattr(response, 'text') else "Failed to generate poem."

# if "selected_mood" not in st.session_state:
#     st.session_state.selected_mood = ""
# if "mood_button_clicked" not in st.session_state:
#     st.session_state.mood_button_clicked = False

# st.title("📝 Mood-Based Poem Generator")

# user_input = st.text_area("Enter your thoughts (optional if selecting mood below):")
# language = st.selectbox("Choose poem language:", ["English", "Hindi", "Telugu"])

# st.markdown("**Or select a mood directly:**")
# cols = st.columns(5)
# mood_labels = ["Happy", "Excited", "Sad", "Angry", "Neutral"]

# for i, label in enumerate(mood_labels):
#     if cols[i].button(label):
#         st.session_state.selected_mood = label.lower()
#         st.session_state.mood_button_clicked = True

# if st.button("Generate Poem"):
#     mood = None

#     if st.session_state.mood_button_clicked:
#         mood = st.session_state.selected_mood
#         st.subheader(f"Selected Mood: {mood.capitalize()}")
#         st.session_state.mood_button_clicked = False 
#     elif user_input.strip():
#         mood = get_emotion(user_input)
#         st.subheader(f"Detected Mood: {mood.capitalize()}")

#     if mood:
#         poem = generate_poem(mood, language)
#         st.markdown(f"### ✨ Poem in {language} for '{mood.capitalize()}' Mood")
#         st.write(poem)
#         st.session_state.selected_mood = ""  
#     else:
#         st.error("Please enter some text or select a mood.")


# # get_emotion("This is the best day of my life.")
# # get_emotion("I feel so alone and unmotivated.")
# # get_emotion("Stop bothering me!")
# # get_emotion("I'm just doing my regular routine.")
# # get_emotion("woah")

import streamlit as st
import torch
from google import genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Emotion detection model
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def get_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return labels[predicted_class]

# Poem generation
def generate_poem(mood, language):
    client = genai.Client(api_key="AIzaSyCxAu00RguxS3mo-b025nF4FtIg1fVvlkY")  # Replace with your actual API key
    prompt = f"Generate a {mood} poem in {language}."
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text if hasattr(response, 'text') else "Failed to generate poem."

# State management
if "selected_mood" not in st.session_state:
    st.session_state.selected_mood = ""
if "mood_button_clicked" not in st.session_state:
    st.session_state.mood_button_clicked = False

# ----------------- CUSTOM CSS -----------------
st.markdown("""
    <style>
    /* Background and font */
    .stApp {
        color: black;
        background: #C890A7;
        font-family: 'Segoe UI', sans-serif;
            
    }
    p{
            color:black;}
    h1 {
        color: #A35C7A;
        text-align: center;
        font-size: 3em;
        margin-bottom: 10px;
    }

    h2, h3 {
        color: black;
        margin-top: 20px;
    }

    .poem-box {
        background-color: #A35C7A;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 16px;
        line-height: 1.6;
        color: black;
    }

    .stButton>button {
        background-color: #FBF5E5;
        color: black;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #ffaa00;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- APP UI -----------------
st.title("📝 Mood-Based Poem Generator")

user_input = st.text_area("Enter your thoughts (optional if selecting mood below):")
language = st.selectbox("Choose poem language:", ["English", "Hindi", "Telugu"])

st.markdown("**Or select a mood directly:**")
cols = st.columns(5)
mood_labels = ["Happy", "Excited", "Sad", "Angry", "Neutral"]

for i, label in enumerate(mood_labels):
    if cols[i].button(label):
        st.session_state.selected_mood = label.lower()
        st.session_state.mood_button_clicked = True

if st.button("Generate Poem"):
    mood = None

    if st.session_state.mood_button_clicked:
        mood = st.session_state.selected_mood
        st.subheader(f"Selected Mood: {mood.capitalize()}")
        st.session_state.mood_button_clicked = False 
    elif user_input.strip():
        mood = get_emotion(user_input)
        st.subheader(f"Detected Mood: {mood.capitalize()}")

    if mood:
        poem = generate_poem(mood, language)
        st.markdown(f"### ✨ Poem in {language} for '{mood.capitalize()}' Mood")
        st.markdown(f"<div class='poem-box'>{poem}</div>", unsafe_allow_html=True)
        st.session_state.selected_mood = ""  
    else:
        st.error("Please enter some text or select a mood.")

