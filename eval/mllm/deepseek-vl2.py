import os
import csv
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


os.environ['CUDA_VISIBLE_DEVICES'] = "5,6,7"
model_path = "/models/deepseek-vl2"


vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto"
).to(torch.bfloat16).eval()

def process_row(row, processor, model, tokenizer, activity_id):
    
    x = row[0]  
    text = row[1]  


    base_path = '/exp/MM-SafetyBench(imgs)'
    img_full_path = os.path.join(base_path, activity_id, 'TYPO', f"{x}.jpg")

    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{text}",
            "images": [img_full_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]


    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)


    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )


    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    if len(row) < 10:
        row.extend([''] * (10 - len(row)))  
    row[9] = answer  
    del inputs_embeds, prepare_inputs, outputs
    torch.cuda.empty_cache()
    return row


input_dir = 'exp/csv_same'  
output_dir = 'exp/typo_response'  


os.makedirs(output_dir, exist_ok=True)


for input_filename in os.listdir(input_dir):
    if input_filename.endswith('.csv'):  
      
        activity_id = os.path.splitext(input_filename)[0]  

        input_filepath = os.path.join(input_dir, input_filename)
        output_filepath = os.path.join(output_dir, input_filename)

        print(f"正在处理文件: {input_filepath}，使用 activity_id: {activity_id}")

        with open(input_filepath, mode='r', encoding='utf-8') as infile, \
             open(output_filepath, mode='w', encoding='utf-8', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)


            for row in reader:
                processed_row = process_row(row, vl_chat_processor, vl_gpt, tokenizer, activity_id)
                writer.writerow(processed_row)

        print(f"处理完成，结果已保存到: {output_filepath}")