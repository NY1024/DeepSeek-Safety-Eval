import os
import pandas as pd
import re

def clean_text(text):

    return re.sub(r'^\d+\.', '', text).strip()

def process_csv_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            input_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)
            

            df = pd.read_csv(input_file, dtype=str)

            df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)
            
            df.to_csv(output_file, index=False)
            print(f"Processed and saved: {output_file}")

input_directory = "/home/elwood/deepseek/cn_safe" 
output_directory = "processed_csvs"  
process_csv_files(input_directory, output_directory)
