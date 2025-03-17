import os
import pandas as pd

def split_csv_columns(input_file, output_dir):
  
    os.makedirs(output_dir, exist_ok=True)
    

    df = pd.read_csv(input_file)
    
    for i, column in enumerate(df.columns, start=1):
        output_file = os.path.join(output_dir, f"{i}.csv")
        df[[column]].to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

input_csv = "/home/elwood/deepseek/cn_safe.csv"  
output_directory = "/home/elwood/deepseek/cn_safe" 
split_csv_columns(input_csv, output_directory)