import os
from openai import AzureOpenAI

from config import Config

def main():
    # Azure OpenAI configurations
    # Load configuration
    config = Config("config.json")
    
    # Create chatbot instance
    client = AzureOpenAI(
        api_key = config.api_key,  
        api_version = config.api_version,
        azure_endpoint = config.api_base
        )

    response = client.chat.completions.create(
        model=config.deployment_name,
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": "Who were the founders of Microsoft?"}
        ]
    )

    # print(response.model_dump_json(indent=2))
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()