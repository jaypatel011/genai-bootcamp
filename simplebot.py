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
    # print("Chat with AI (type 'quit' to exit)")
    # print("---------------------------------")

    # while True:
    #     # Get user input
    #     user_input = input("You: ")
        
    #     # Check if user wants to quit
    #     if user_input.lower() == 'quit':
    #         print("Goodbye!")
    #         break

    #     try:
    #         # Create chat completion
    #         response = openai.chat.completions.create(
    #             model=config.deployment_name,
    #             messages=[
    #                 {"role": "user", "content": user_input}
    #             ],
    #             temperature=0.7,
    #             max_tokens=800
    #         )

    #         # Print the response
    #         print("\nAI:", response.choices[0].message.content)
    #         print()

    #     except Exception as e:
    #         print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()