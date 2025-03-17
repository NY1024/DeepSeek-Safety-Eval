import os
import csv
import torch
import numpy as np
import PIL.Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


model_path = "/DS/Janus-Pro-7B"


vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


@torch.inference_mode()
def generate1(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    output_name: str,              
    output_path: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
   
):
    
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

   
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)


    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs(output_path, exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join(output_path, f"{output_name}_img_{i}.jpg")
        PIL.Image.fromarray(visual_img[i]).save(save_path)
    print(f"图片已保存，输出前缀：{output_name}")


def process_csv(csv_path: str,output_path:str):

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            prompt_text = row.get("prompt")
            if prompt_text is None:
                print(f"第 {idx} 行未找到 'prompt' 字段，跳过。")
                continue

            conversation = [
                {"role": "User", "content": prompt_text},
                {"role": "Assistant", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt=""
            )
            full_prompt = sft_format + vl_chat_processor.image_start_tag

            output_name = f"sample_{idx}"
            print(f"正在生成第 {idx} 个 prompt 对应的图像，输出名称：{output_name}")
            generate1(
                vl_gpt,
                vl_chat_processor,
                full_prompt,
                output_name=output_name,
                output_path=output_path,
            )


if __name__ == "__main__":
    csv_file = "/output_files/sexual.csv"  
    process_csv(csv_file,"/result/sexual")  
    csv_file = "/output_files/hate.csv"  
    process_csv(csv_file,"/result/hate") 
    csv_file = "/output_files/violence.csv"  
    process_csv(csv_file,"/result/violence")  
    csv_file = "/output_files/illegal-activity.csv"  
    process_csv(csv_file,"/result/illegal-activity")   
    csv_file = "/output_files/self-harm.csv"  
    process_csv(csv_file,"/result/self-harm") 
    csv_file = "/output_files/shocking.csv"  
    process_csv(csv_file,"/result/shocking")  
    csv_file = "/output_files/harassment.csv"  
    process_csv(csv_file,"/result/harassment") 