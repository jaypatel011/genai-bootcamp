import os
import streamlit as st
from openai import AzureOpenAI
from typing import List, Dict
import json
import ast
from config import Config
from scipy.spatial import distance
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

class Chatbot:
    def __init__(self, config: Config):
        self.config = config
        self.setup_azure_openai()
        self.conversation_history: List[Dict] = []
        self.initialize_conversation()
        self.load_embeddings_from_csv('faqs_embeddings.csv')

    def setup_azure_openai(self):
        self.client = AzureOpenAI(
        api_key = self.config.api_key,  
        api_version = self.config.api_version,
        azure_endpoint = self.config.api_base
        )
        
        self.embeddingsClient = AzureOpenAI(
        api_key = self.config.embedding_api_key,  
        api_version = self.config.embedding_api_version,
        azure_endpoint = self.config.embedding_api_base
        )
        
        

    def initialize_conversation(self):
        self.conversation_history = [
            {"role": "system", "content": self.config.system_message}
        ]

    def generate_response(self, user_input: str) -> str:
        try:
            context_df = self.retrieve_relevant_context(user_input, top_k=2)
            context = "\n".join([f"Q: {row.question}\nA: {row.answer}" for _, row in context_df.iterrows()])
            self.conversation_history.append({"role": "system", "content": f"Answer using ONLY this context:\n{context}"})
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Get response from Azure OpenAI using embeddings as context
            response = self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=self.conversation_history,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Extract assistant's response
            assistant_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            return f"Error: {str(e)}"

    def load_embeddings_from_csv(self, fileName):
        # Load data and embeddings
        self.df = pd.read_csv(fileName)
        # Convert the string representation of embeddings back to lists of floats
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

    
    def get_embedding(self, text):
        response = self.embeddingsClient.embeddings.create(
                model=self.config.embedding_deployment_name,
                input=text,
            )
        return response.data[0].embedding

    def retrieve_relevant_context(self, query, top_k=2):
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Calculate similarity scores
        self.df["similarity"] = self.df["embedding"].apply(
            lambda x: cosine_similarity([query_embedding], [x])[0][0]
        )
        # Return top matches
        return self.df.sort_values("similarity", ascending=False).head(top_k)

    def clear_conversation(self):
        self.initialize_conversation()
        return "", ""

def create_streamlit_interface(chatbot: Chatbot):
    st.title(chatbot.config.title)
    st.write(chatbot.config.description)
    
    # Store conversation history in session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        
    # Create a container for the chat history
    with st.container():
        display_conversation()
    
    st.text_input("You:", key="user_input", on_change=lambda: send_message(chatbot))
    
    if st.button("Clear Conversation"):
        chatbot.clear_conversation()
        st.session_state.conversation_history = []  # Clear the session state history
        st.session_state.last_input = ""  # Clear the last input (optional)

# Callback function for sending the message
def send_message(chatbot: Chatbot):
    user_input = st.session_state.user_input
    if user_input:
        # Add user input to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Get response from the chatbot
        response = chatbot.generate_response(user_input)
        
        # Add assistant response to conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        
        # Clear input after sending
        st.session_state.user_input = ""  # Clear the last input
          
def display_conversation():
    for chat in st.session_state.conversation_history:
        if chat["role"] == "user":
            # Right align user's message
            st.markdown(
                f"<div style='text-align: right;'><span style='font-size: 18px;'>{USER_AVATAR} {chat['content']}</span></div>",
                unsafe_allow_html=True
            )
        elif chat["role"] == "assistant":
            # Left align assistant's message
            st.markdown(
                f"<div style='text-align: left; margin-bottom:20px; color:#00FFFF;'><span style='font-size: 20px;'>{BOT_AVATAR} {chat['content']}</span></div>",
                unsafe_allow_html=True
            )
            
def main():
    # Load configuration
    config = Config("config.json")
    
    # Create chatbot instance
    chatbot = Chatbot(config)
    
    # Create and launch Gradio interface
    create_streamlit_interface(chatbot)

if __name__ == "__main__":
    main()
