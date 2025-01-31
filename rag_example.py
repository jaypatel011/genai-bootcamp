# Using Embeddings and Azure OpenAI to Answer Questions from Custom Data
# Step 1: Install libraries
# !pip install openai python-dotenv pandas scikit-learn

# Step 2: Import modules
import os
from openai import AzureOpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

# Step 3: Load Azure OpenAI credentials
# Load configuration
config = Config("config.json")
    
chat_client = AzureOpenAI(
    api_key = config.api_key,  
    api_version = config.api_version,
    azure_endpoint = config.api_base
    )
        
embeddings_client = AzureOpenAI(
    api_key = config.embedding_api_key,  
    api_version = config.embedding_api_version,
    azure_endpoint = config.embedding_api_base
    )

# Step 4: Create a CSV with custom data (e.g., company FAQs)
csv_data = """
question,answer
"What is the return policy?","Returns are accepted within 30 days with a receipt."
"How do I reset my password?","Visit the login page and click 'Forgot Password' to reset."
"What's the warranty period?","All products come with a 2-year manufacturer warranty."
"Do you offer international shipping?","Yes, we ship to 50+ countries with extra fees."
"What payment methods do you accept?","We accept credit cards, PayPal, and Apple Pay."
"""
with open("faqs.csv", "w") as f:
    f.write(csv_data)

# Step 5: Load data and generate embeddings
df = pd.read_csv("faqs.csv")
def get_embedding(text):
    response = embeddings_client.embeddings.create(
        input=text,
        model=config.embedding_deployment_name
    )
    return response.data[0].embedding
# Generate embeddings for questions
df["embedding"] = df["question"].apply(get_embedding)
df.to_csv('faqs_embeddings.csv', index=False)

# Step 6: Build a retrieval system to find relevant answers
def retrieve_relevant_context(query, df, top_k=2):
    # Get query embedding
    query_embedding = get_embedding(query)
    # Calculate similarity scores
    df["similarity"] = df["embedding"].apply(
        lambda x: cosine_similarity([query_embedding], [x])[0][0]
    )
    # Return top matches
    return df.sort_values("similarity", ascending=False).head(top_k)

# Step 7: Create a RAG (Retrieval-Augmented Generation) function
def ask_llm(query, df):
    # Retrieve relevant context
    context_df = retrieve_relevant_context(query, df)
    context = "\n".join([f"Q: {row.question}\nA: {row.answer}" for _, row in context_df.iterrows()])
    # Generate answer using LLM
    response = chat_client.chat.completions.create(
        model=config.deployment_name, # Use your Azure deployment name
        messages=[
            {"role": "system", "content": f"Answer using ONLY this context:\n{context}"},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# Step 8: Test with creative examples
# Example 1: Direct match
query1 = "How long is the warranty period?"
answer1 = ask_llm(query1, df)
print(f"\nQ: {query1}\nA: {answer1}")
# Example 2: Paraphrased question
query2 = "Can I get my money back after 3 weeks?"
answer2 = ask_llm(query2, df)
print(f"\nQ: {query2}\nA: {answer2}")
# Example 3: Implicit context
query3 = "Do you support payments via digital wallets?"
answer3 = ask_llm(query3, df)
print(f"\nQ: {query3}\nA: {answer3}")
# Example 4: Noisy query
query4 = "hey how 2 change pwd? pls help"
answer4 = ask_llm(query4, df)
print(f"\nQ: {query4}\nA: {answer4}")