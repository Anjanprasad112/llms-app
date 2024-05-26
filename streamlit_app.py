from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
from fuzzywuzzy import fuzz

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Load Q&A dataset
@st.cache_data
def load_qa_dataset():
    df = pd.read_csv("medquad.csv")
    return df

qa_dataset = load_qa_dataset()

# Function to get Gemini response
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Function to find expert advice based on the user's query
def find_expert_advice(query):
    # Calculate similarity scores between the query and each question in the dataset
    match_threshold = 70  # Adjust as needed
    match_scores = qa_dataset['question'].apply(lambda x: fuzz.token_set_ratio(query.lower(), x.lower()))
    max_score_index = match_scores.idxmax()
    max_score = match_scores[max_score_index]
    # If the maximum score is above the threshold, return the corresponding answer
    if max_score >= match_threshold:
        return qa_dataset.loc[max_score_index, 'answer']
    else:
        return None

# Initialize Streamlit app
# st.set_page_config(page_title="AiDvice-Medical")
st.header("AiDvice-Medical")

# Display disclaimer
st.sidebar.subheader("Disclaimer:")
st.sidebar.write("This application provides general health information and is for educational purposes only. It should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")


# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input field for user query
input_query = st.text_input("Input: ", key="input")

# Button to submit the query
submit_button = st.button("Ask the question")

# If user submits a query
if submit_button and input_query:
    # Get response from Gemini Pro model
    expert_advice = find_expert_advice(input_query)
    if expert_advice:
        # Display expert advice
        st.subheader("Expert Advice:")
        st.write(expert_advice)
        st.session_state['chat_history'].append(("Expert", expert_advice))
    response = get_gemini_response(input_query)
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", input_query))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))
    


