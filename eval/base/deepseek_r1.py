import os
import pandas as pd
import re
from openai import OpenAI

def clean_text(text):
    return re.sub(r'^\d+\.', '', text).strip()

def query_model(query, client):
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": query}],
        max_tokens=2048,
        stream=False
    )
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    return f"{reasoning_content}@{content}"

def process_csv_files(input_dir, output_dir, api_key, base_url):
    os.makedirs(output_dir, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=base_url)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            input_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)

            df = pd.read_csv(input_file, dtype=str)

            processed_data = []

            for index, row in df.iterrows():
                for col in df.columns:
                    original_text = row[col]
                    if isinstance(original_text, str):
                        cleaned_text = clean_text(original_text)
                        model_response = query_model(cleaned_text, client)
                        processed_data.append([original_text, model_response])
                print(f"Processing {file_name}, line {index + 1}")

            result_df = pd.DataFrame(processed_data, columns=["Original Text", "Model Response"])
            result_df.to_csv(output_file, index=False)
            print(f"Processed and saved: {output_file}")


input_directory = "/home/elwood/deepseek/processed_csvs"  
output_directory = "/home/elwood/deepseek/r1_response"  
api_key = "sk-*"  
base_url = "https://api.deepseek.com" 
process_csv_files(input_directory, output_directory, api_key, base_url)
