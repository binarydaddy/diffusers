# Current task is to run StableDiffusion2 on COCO 2017, and compare FIDs.
# Thankfully, StableDiffusion2 trained on both v-prediction and score prediction.
# We can compare the performance, along with v-pred -> score.
# Furthermore, we can try using different sampler (as noted by SiT paper), in generating samples.

# coco2017 captions have 5 caption per sample.
import torch
import diffusers
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, HeunDiscreteScheduler

def extract_captions_from_coco():
    '''
        coco2017 validation set has 5 captions per validation sample.
        There are total of 5000 images, and thus 25000 captions.
        Saved under "sit/coco2017_val_captions.txt"
    '''
    import json
    import os
    
    val_anno_path = "/data2/coco2017/annotations/captions_val2017.json"

    with open(val_anno_path, 'r')as f:
        val_anno = json.load(f)

    full_captions = []

    for annotation in val_anno['annotations']:
        caption = annotation['caption']
        full_captions.append(caption)

    if not os.path.exists('coco2017_val_captions.txt'):
        with open('coco2017_val_captions.txt', 'w') as f:
            for cap in full_captions:
                context = '\n' + cap
                f.writelines(context)
        print(f"Done writing")

def val_coco_sit(nfe=1000):
    
    # Initialize StableDiffusion2 model
    # stable-diffusion-2-base (512-base-ema.ckpt) is trained with score objective
    # stable-diffusion-2 (768-v-ema.ckpt) is trained with v-prediction objective
    base_model_id = "stabilityai/stable-diffusion-2-base"
    v_model_id = "stabilityai/stable-diffusion-2"

    base_model = StableDiffusionPipeline.from_pretrained(base_model_id)
    v_model = StableDiffusionPipeline.from_pretrained(v_model_id)

    scheduler = SiTScheduler().from_pretrained()

    # base model is solved through sde, and v model is solved through ode
    

    
    return


if __name__ == "__main__":
    # extract_captions_from_coco()

    val_coco_sit()1