# Current task is to run StableDiffusion2 on COCO 2017, and compare FIDs.
# Thankfully, StableDiffusion2 trained on both v-prediction and score prediction.
# We can compare the performance, along with v-pred -> score.
# Furthermore, we can try using different sampler (as noted by SiT paper), in generating samples.

# coco2017 captions have 5 caption per sample.
import torch
import diffusers
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, HeunDiscreteScheduler, SiTScheduler, DDIMScheduler
import enum

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = SiTScheduler.from_pretrained(model_id, subfolder="scheduler", solver_type="ode", path_type="gvp")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=250).images[0]
    
image.save("astronaut_rides_horse_ddim.png")