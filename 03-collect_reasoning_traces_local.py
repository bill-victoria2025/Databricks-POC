# %%
from databricks.connect import DatabricksSession
from dotenv import load_dotenv
import pandas as pd
import requests
import os
from datetime import datetime
from tqdm import tqdm

load_dotenv()


spark = (
    DatabricksSession.builder.profile("DEFAULT").remote(serverless=True).getOrCreate()
)

df = spark.sql("SELECT * FROM main.sgfs.car_forum_train_canon_vw").toPandas()

# %%
prompt_template = """
Using the <TASK> , <QUERY> and <RESPONSE> scrapped from an automotive forum, 
extract the meaningful gist of the content from the <RESPONSE> and construct the best response for the <QUERY>.  
This new generated response should go inside <ANSWER> </ANSWER> tags. 
Structure it in a way thats most useful to the person posting the <QUERY> just like how someone will post a response in an online forum. 
Don't include finishing signatures or human names.

<TASK>
{task}
</TASK>

<QUERY>
{query}
</QUERY>

<RESPONSE>
{response}
</RESPONSE>
"""

# %%
# Ollama API settings
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

# Function to call the Ollama API
def get_ollama_response(prompt, model="llama3"):
    """
    Send a prompt to Ollama API and get the response
    
    Args:
        prompt (str): The prompt to send to Ollama
        model (str): The model to use (default: llama3)
        
    Returns:
        str: The generated response or None if there was an error
    """
    try:
        # Construct the API endpoint
        api_endpoint = f"{OLLAMA_API_URL}/api/generate"
        
        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024  # Similar to max_tokens
            }
        }
        
        # Make the API request
        response = requests.post(api_endpoint, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        response_data = response.json()
        
        # Extract the generated text
        if "response" in response_data:
            return response_data["response"]
        else:
            print("Unexpected response format from Ollama API")
            return None
            
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None

# Process each row in the dataframe
def process_dataframe(df, model="deepseek-r1:8b", sample_size=None):
    # Optionally sample a subset of the dataframe
    if sample_size and sample_size < len(df):
        processed_df = df.sample(sample_size, random_state=42)
    else:
        processed_df = df.copy()
    
    # Add columns for the API responses and prompts
    processed_df['response'] = None
    processed_df['prompt'] = None
    
    # Process each row
    for idx, row in tqdm(processed_df.iterrows(), total=len(processed_df)):
        # Format the prompt using the template
        formatted_prompt = prompt_template.format(
            task=row['instruction'],
            query=row['input'],
            response=row['output']
        )
        
        # Store the formatted prompt
        processed_df.at[idx, 'prompt'] = formatted_prompt
        
        # Call the Ollama API
        api_response = get_ollama_response(formatted_prompt, model=model)
        processed_df.at[idx, 'response'] = api_response
    
    return processed_df

# Example usage
# Uncomment and modify as needed
result_df = process_dataframe(df, model="deepseek-r1:8b", sample_size=len(df))
today = datetime.now().strftime("%Y-%m-%d")
result_df.to_csv(f'reasoning_model_responses_{today}.csv', index=False)

# %%
result_df.head()

# %%


# %%
