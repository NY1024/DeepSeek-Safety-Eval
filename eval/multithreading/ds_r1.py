import os
import pandas as pd
import re
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

def clean_text(text):
    return re.sub(r'^\d+\.', '', text).strip()

def query_model(query, client, max_retries=5):

    retry_count = 0
    base_delay = 1  

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model="deepseek-r1", 
                messages=[{'role': 'user', 'content': query}]
            )
            reasoning_content = completion.choices[0].message.reasoning_content
            content = completion.choices[0].message.content
            return f"{reasoning_content}@{content}"
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Max retries reached. Failed to process query: {query}")
                raise
            delay = base_delay * (2 ** (retry_count - 1))  
            print(f"Error occurred: {e}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
            time.sleep(delay)

def process_csv_file(input_dir, output_dir, api_key, base_url, file_name):

    os.makedirs(output_dir, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    input_file = os.path.join(input_dir, file_name)
    output_file = os.path.join(output_dir, file_name)
    
    if os.path.exists(input_file):

        df = pd.read_csv(input_file, dtype=str)

        processed_data = []
        for index, row in df.iterrows():
            for col in df.columns:
                original_text = row[col]
                if isinstance(original_text, str):
                    cleaned_text = clean_text(original_text)
                    try:
                        model_response = query_model(cleaned_text, client)
                        processed_data.append([original_text, model_response])
                    except Exception as e:
                        print(f"Failed to process text: {cleaned_text}. Error: {e}")
                        processed_data.append([original_text, "ERROR"])
            print(f"Processing {file_name}, line {index + 1}")

        result_df = pd.DataFrame(processed_data, columns=["Original Text", "Model Response"])
        result_df.to_csv(output_file, index=False)
        print(f"Processed and saved: {output_file}")
    else:
        print(f"File not found: {input_file}")

def main():
  
    input_directory = "/home/elwood/ds_new/naive_dataset" 
    output_directory = "/home/elwood/ds_new/r1_naive_response"  
    target_files = ["11.csv", "12.csv", "13.csv", "14.csv", "15.csv"]  
    api_keys = ["sk", "sk", "sk", "sk", "sk"]  
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  


    with ThreadPoolExecutor(max_workers=len(target_files)) as executor:
        for i, file_name in enumerate(target_files):
            api_key = api_keys[i]  
            executor.submit(
                process_csv_file,
                input_directory,
                output_directory,
                api_key,
                base_url,
                file_name
            )

if __name__ == "__main__":
    main()