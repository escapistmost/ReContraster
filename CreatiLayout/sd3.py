import os
import torch
from diffusers import StableDiffusion3Pipeline

model_path = os.getenv("SD3_MODEL_PATH")
if not model_path:
    raise RuntimeError("请设置 SD3_MODEL_PATH")
pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    # "A hyper-realistic depiction of a clear blue ocean under a bright sunny sky with white clouds, a tiny sandy island in the middle of the water with a single green sapling growing on it, symbolizing hope and fragile life. The scene is serene and detailed, with realistic lighting and a vibrant, natural color palette, evoking a sense of purity and calmness.",
    "Realistic natural landscape with a meandering river flowing through dense, lush forests and rolling hills. Vibrant, rich colors create a lively scene with sunlight filtering through the canopy and gentle reflections on the water. Detailed and lifelike, showcasing a serene and picturesque atmosphere.",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images
for i,k in enumerate(image):
    k.save(f'./output/sd3/{i}.png')
