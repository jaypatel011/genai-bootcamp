import json

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