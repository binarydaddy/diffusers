# Current task is to run StableDiffusion2 on COCO 2017, and compare FIDs.
# Thankfully, StableDiffusion2 trained on both v-prediction and score prediction.
# We can compare the performance, along with v-pred -> score.
# Furthermore, we can try using different sampler (as noted by SiT paper), in generating samples.

# coco2017 captions have 5 caption per sample.
import torch
import diffusers
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, HeunDiscreteScheduler, SiTScheduler, DDIMScheduler
import enum

def main(args):
    model_id = "stabilityai/stable-diffusion-2"

    torch.manual_seed(0)

    # Use the Euler scheduler here instead
    if args.solver_type == "ddim":
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    else:
        scheduler = SiTScheduler.from_pretrained(model_id, subfolder="scheduler", 
                                                solver_type=args.solver_type, 
                                                path_type="gvp",
                                                diffusion_form=args.diffusion_form)    
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    
    image = pipe(args.prompt, num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale).images[0]        
    image.save(f"test_out/test_{args.solver_type}_{args.diffusion_form}_diffusion_{args.inference_steps}_{args.guidance_scale}.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-type", type=str, choices=["sde", "ode", "ddim"], default="sde")
    parser.add_argument("--inference-steps", type=int, default=500)
    parser.add_argument("--diffusion-form", choices=["linear", "constant", "sigma"], type=str, default="constant")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars")
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    args = parser.parse_args()

    main(args)