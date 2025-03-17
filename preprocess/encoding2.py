import os
import chardet

def detect_encoding(file_path, sample_size=100000):
    with open(file_path, "rb") as f:
        rawdata = f.read(sample_size)  
        result = chardet.detect(rawdata)
    return result["encoding"]

def convert_to_utf8(input_file, output_file):
    detected_encoding = detect_encoding(input_file)
    print(f"文件 {input_file} 检测到的编码: {detected_encoding}")

    if detected_encoding is None:
        print(f"文件 {input_file} 无法检测编码，默认使用 ISO-8859-1 进行转换...")
        detected_encoding = "ISO-8859-1"

    with open(input_file, "r", encoding=detected_encoding, errors="ignore") as f:
        content = f.read()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ 文件已转换为 UTF-8: {output_file}")

def batch_convert_csv(input_folder, output_folder):
    """批量转换文件夹中的所有 CSV 文件为 UTF-8"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):  
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)

            try:
                convert_to_utf8(input_file, output_file)
            except Exception as e:
                print(f"❌ 处理 {input_file} 时出错: {e}")

input_directory = "/home/elwood/ds_new/todo"  
output_directory = "/home/elwood/ds_new/todo1"

batch_convert_csv(input_directory, output_directory)
