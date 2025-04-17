"""

해당 내용은 Colab 에서 실행되는 걸 전제로 만들어짐

"""

# 📌 1. 필요한 라이브러리 설치
# !pip install diffusers transformers accelerate safetensors

# 📌 2. 라이브러리 임포트
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
import json
import os

# 📌 3. 모델 및 LoRA 로드
# SDXL 모델 로드
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

# 스케줄러 설정
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 지브리 스타일 LoRA 로드
pipe.load_lora_weights("ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style", weight_name="Studio Ghibli style.safetensors", adapter_name="Studio Ghibli style")

# LoRA 활성화
pipe.set_adapters(["Studio Ghibli style"], adapter_weights=[2.0])

# 📌 4. 변환된 프롬프트 로드
with open("converted_prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

# 📌 5. 이미지 생성 및 저장
output_dir = "ghibli_images"
os.makedirs(output_dir, exist_ok=True)

for idx, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        negative_prompt="nsfw, low quality, blurry",
        width=768,
        height=768,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]
    image.save(os.path.join(output_dir, f"cut_{idx+1}.png"))
    print(f"컷 {idx+1} 이미지 저장 완료: {os.path.join(output_dir, f'cut_{idx+1}.png')}")
