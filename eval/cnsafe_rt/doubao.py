import os
import pandas as pd
import requests
import json
import time

API_SECRET_KEY = "**"
BASE_URL = "**"

# 新 API 的调用函数，加入容错机制
def gpt4_shadow(messages, max_retries=5, retry_delay=2):

    payload = json.dumps({
        "model": "Doubao-1.5-pro-32k-250115",
        "messages": messages,
        "stream": False,
        "temperature": 0.9,
        "max_tokens": 2048,
        "top_p": 0.5,
        "top_k": 0,
        "repetition_penalty": 1.05,
        "num_beams": 1,
        "user": "xx",
        "content_filter": False
    })
    headers = {
        'Authorization': API_SECRET_KEY,
        'Content-Type': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(BASE_URL, headers=headers, data=payload)

            if response.status_code == 200:

                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"API 调用失败，状态码: {response.status_code}, 信息: {response.text}")
                if attempt < max_retries - 1:
                    print(f"将在 {retry_delay} 秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"API 调用失败，状态码: {response.status_code}, 信息: {response.text}")
        except Exception as e:
            print(f"调用失败: {e}，将在 {retry_delay} 秒后重试 ({attempt + 1}/{max_retries})...")
            time.sleep(retry_delay)

    raise Exception("API 调用多次失败，已达到最大重试次数")

def process_csv(input_file, output_file):

    df = pd.read_csv(input_file)

    results = []
    

    for _, row in df.iterrows():
        query = str(row[0])  

        final_prompt = query

        try:
            messages = [{"role": "user", "content": final_prompt}]
            result = gpt4_shadow(messages) 
        except Exception as e:
            print(f"API 调用失败: {e}")
            result = '{"result": "error", "reason": "API call failed"}'
        

        results.append(result)
    

    df["result"] = results

    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")
件
def process_folder(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            
            print(f"正在处理文件: {input_file}")
            process_csv(input_file, output_file)

if __name__ == "__main__":
    input_folder = "/home/beihang/yzh/new1/attack_dataset"  
    output_folder = "/home/beihang/yzh/new1/attack_doubao_response"  
    
    process_folder(input_folder, output_folder)