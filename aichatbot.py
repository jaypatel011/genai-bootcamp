import os
import gradio as gr
import openai
from typing import List, Dict
import json

# Configuration class to store all settings
class Config:
    def __init__(self, config_file: str = "config.json"):
        self.load_config(config_file)

    def load_config(self, config_file: str):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Azure OpenAI settings
            self.api_type = config.get("api_type", "azure")
            self.api_base = config.get("api_base", "")
            self.api_version = config.get("api_version", "2023-05-15")
            self.api_key = config.get("api_key", "")
            self.deployment_name = config.get("deployment_name", "")
            
            # Chat settings
            self.temperature = config.get("temperature", 0.7)
            self.max_tokens = config.get("max_tokens", 800)
            self.system_message = config.get("system_message", 
                "You are a helpful assistant that provides accurate and concise responses.")
            
            # Gradio UI settings
            self.theme = config.get("theme", "default")
            self.title = config.get("title", "AI Chatbot")
            self.description = config.get("description", "Chat with AI Assistant")
            self.examples = config.get("examples", [])
            
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default values.")
            self.set_defaults()

    def set_defaults(self):
        # Set default values if config file is not found
        self.api_type = "azure"
        self.api_base = ""
        self.api_version = "2023-05-15"
        self.api_key = ""
        self.deployment_name = ""
        self.temperature = 0.7
        self.max_tokens = 800
        self.system_message = "You are a helpful assistant that provides accurate and concise responses."
        self.theme = "default"
        self.title = "AI Chatbot"
        self.description = "Chat with AI Assistant"
        self.examples = []

class Chatbot:
    def __init__(self, config: Config):
        self.config = config
        self.setup_azure_openai()
        self.conversation_history: List[Dict] = []
        self.initialize_conversation()

    def setup_azure_openai(self):
        openai.api_type = self.config.api_type
        openai.api_base = self.config.api_base
        openai.api_version = self.config.api_version
        openai.api_key = self.config.api_key

    def initialize_conversation(self):
        self.conversation_history = [
            {"role": "system", "content": self.config.system_message}
        ]

    def generate_response(self, user_input: str) -> str:
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Get response from Azure OpenAI
            response = openai.ChatCompletion.create(
                engine=self.config.deployment_name,
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