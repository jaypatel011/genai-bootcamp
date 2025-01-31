import os
import gradio as gr
from openai import AzureOpenAI
from typing import List, Dict
import json
import ast
from config import Config
from scipy.spatial import distance
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
            print(user_input)
            context_df = self.retrieve_relevant_context(user_input, top_k=2)
            print(context_df)
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
        print(query_embedding)
        # Calculate similarity scores
        self.df["similarity"] = self.df["embedding"].apply(
            lambda x: cosine_similarity([query_embedding], [x])[0][0]
        )
        # Return top matches
        return self.df.sort_values("similarity", ascending=False).head(top_k)

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
    # Load configuration
    config = Config("config.json")
    
    # Create chatbot instance
    chatbot = Chatbot(config)
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(chatbot)
    interface.launch(share=True)

if __name__ == "__main__":
    main()
