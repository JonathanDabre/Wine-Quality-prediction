import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def query_gemini(prompt):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Hey you are chatbot in an Wine related application, the question is : {prompt}. (If you feel question is related to wines then only answer or else say 'The question doesn't seem to be related to Wines')"
                    }
                ]
            }
        ]
    }

    response = requests.post(api_url, headers=headers, json=data, params={"key": api_key})
    
    # Debugging: Print the response content
    print("Response Status Code:", response.status_code)
    print("Response Content:", response.text)  # This will show the full response for debugging

    if response.status_code == 200:
        # Attempt to parse the response and extract the text
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except KeyError as e:
            return f"Error: Missing key in response - {e}"
    else:
        return f"Error: {response.text}"

# Chatbot UI implementation
def chatbot_page():
    st.title("Gemini AI Chatbot")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("You: ", "")

    if st.button("Send"):
        if user_input:
            # Append the user input to chat history
            st.session_state.chat_history.append({"user": user_input})

            # Query the Gemini API
            bot_response = query_gemini(user_input)

            # Append the bot response to chat history
            st.session_state.chat_history.append({"bot": bot_response})

    # Display chat history
    for chat in st.session_state.chat_history:
        if "user" in chat:
            st.write(f"You: {chat['user']}")
        if "bot" in chat:
            st.write(f"Bot: {chat['bot']}")
