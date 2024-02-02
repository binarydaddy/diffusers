# from diffusers.pipelines import TestPipeline

# a = TestPipeline()

from diffusers.pipelines import StableDiffusionPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler

base_model_id = "stabilityai/stable-diffusion-2-base"
v_model_id = "stabilityai/stable-diffusion-2"

# v_model = StableDiffusionPipeline.from_pretrained(v_model_id)
v_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base_model_id, subfolder="scheduler")

print(f"{v_scheduler.config}")

