import os
import pandas as pd
from openai import OpenAI

client = OpenAI(

    api_key="*",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def call_model(messages, max_retries=5, retry_delay=2):

    for attempt in range(max_retries):
        try:
            reasoning_content = ""  
            answer_content = ""     
            is_answering = False   
            completion = client.chat.completions.create(
                model="qwq-32b", 
                messages=messages,
                stream=True,
            )

            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content != "" and not is_answering:
                        is_answering = True
                    answer_content += delta.content

            return f"{reasoning_content}@{answer_content}"

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
            result = call_model(messages)  
        except Exception as e:
            print(f"API 调用失败: {e}")
            result = '{"result": "error", "reason": "API call failed"}'
        

        results.append(result)
    

    df["result"] = results

    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")

def process_folder(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            
            print(f"正在处理文件: {input_file}")
            process_csv(input_file, output_file)


if __name__ == "__main__":
    input_folder = "/home/beihang/yzh/new_dataset/attack_dataset1"  
    output_folder = "/home/beihang/yzh/new1/attack_qwq_response"  
    
    process_folder(input_folder, output_folder)