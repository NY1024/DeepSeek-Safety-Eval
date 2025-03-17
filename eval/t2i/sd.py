import torch
import pandas as pd
import os
from diffusers import StableDiffusion3Pipeline
script_dir = "/SD"


pipe = StableDiffusion3Pipeline.from_pretrained(
    "ckpts/sd/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16
).to("cuda")


csv_files = ["/SD/output_files/harassment.csv", "/SD/output_files/hate.csv", "SD/output_files/illegal-activity.csv", "SD/output_files/self-harm.csv", "/output_files/sexual.csv", "/output_files/shocking.csv", "/SD/output_files/violence.csv"]


for csv_file in csv_files:
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found, skipping...")
        continue  

 
    df = pd.read_csv(csv_file)

   
    category_name = os.path.splitext(csv_file)[0] 
    output_dir = os.path.join(script_dir, f"{category_name}Stable")  
    os.makedirs(output_dir, exist_ok=True)


    for index, row in df.iterrows():
        image_id = row["id"]
        prompt = row["prompt"]

        try:
           
            image = pipe(
                prompt,
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images[0]

          
            image_path = os.path.join(output_dir, f"{image_id}.png")
            image.save(image_path)

            print(f"Saved: {image_path}")
        except Exception as e:
            print(f"Error processing {csv_file} (id: {image_id}): {e}")

print("All images generated successfully.")
