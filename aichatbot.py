import os
import gradio as gr
from openai import AzureOpenAI
from typing import List, Dict
import json
from config import Config
from sentence_transformers import SentenceTransformer
import pandas as pd

class Chatbot:
    def __init__(self, config: Config):
        self.config = config
        self.setup_azure_openai()
        self.conversation_history: List[Dict] = []
        self.initialize_conversation()

    def setup_azure_openai(self):
        self.client = AzureOpenAI(
        api_key = self.config.api_key,  
        api_version = self.config.api_version,
        azure_endpoint = self.config.api_base
        )

    def initialize_conversation(self):
        self.conversation_history = [
            {"role": "system", "content": self.config.system_message}
        ]

    def generate_response(self, user_input: str) -> str:
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Generate embeddings for user input
            model = SentenceTransformer('all-MiniLM-L6-v2')
            user_input_embedding = model.encode(user_input)

            # Search embeddings and provide to LLM as context
            closest_embedding_id = self.find_closest_embedding(user_input_embedding)
            context_embedding = self.embeddings.get(closest_embedding_id, [])

            # Get response from Azure OpenAI using embeddings as context
            response = self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=self.conversation_history + [{"role": "embedding", "content": context_embedding}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Extract assistant's response
            assistant_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            return f"Error: {str(e)}"

    def load_embeddings_from_csv(self, csv_file: str):
        """Load data from a CSV file and generate embeddings."""
        data = pd.read_csv(csv_file)
        self.embeddings = {}
        model = SentenceTransformer('all-MiniLM-L6-v2')
        for index, row in data.iterrows():
            self.embeddings[row['id']] = model.encode(row['text'])

    def clear_conversation(self):
        self.initialize_conversation()
        return "", ""

def create_gradio_interface(chatbot: Chatbot):
    with gr.Blocks(theme=chatbot.config.theme, title=chatbot.config.title) as interface:
        gr.Markdown(f"# {chatbot.config.title}")
        gr.Markdown(chatbot.config.description)

        chatbot_interface = gr.Chatbot()
        msg = gr.Textbox(label="Type your message here...")
        clear = gr.Button("Clear Conversation")

        def user_message(message, history):
            if message.strip() == "":
                return "", history
            
            response = chatbot.generate_response(message)
            history.append((message, response))
            return "", history

        msg.submit(user_message, [msg, chatbot_interface], [msg, chatbot_interface])
        clear.click(chatbot.clear_conversation, outputs=[msg, chatbot_interface])

        if chatbot.config.examples:
            gr.Examples(
                examples=chatbot.config.examples,
                inputs=msg
            )

    return interface

def main():
    # Example usage of loading embeddings
    chatbot.load_embeddings_from_csv('path_to_your_csv.csv')
    # Load configuration
    config = Config("config.json")
    
    # Create chatbot instance
    chatbot = Chatbot(config)
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(chatbot)
    interface.launch(share=True)

if __name__ == "__main__":
    main()
