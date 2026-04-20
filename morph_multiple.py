import argparse
import os

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)

from arc2face import CLIPTextModelWrapper, project_face_embs, image_align

from gdl.utils.FaceDetector import FAN
from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import preprocess_for_emoca

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import random


# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda:0"
dtype = torch.float16

# Load face detection and recognition package
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load pipeline
base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=dtype
)
unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=dtype
)
controlnet = ControlNetModel.from_pretrained(
    'models', subfolder="controlnet", torch_dtype=dtype
)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)

# load and disable LCM
pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipeline.disable_lora()

# Load Emoca
face_detector = FAN()
path_to_models = "external/emoca/assets/EMOCA/models"
model_name = 'EMOCA_v2_lr_mse_20'
mode = 'detail'
emoca_model, conf = load_model(path_to_models, model_name, mode)
emoca_model.to(device)
emoca_model.eval()

def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
  '''
  Spherical linear interpolation
  Args:
    v0: Starting vector
    v1: Final vector
    t: Float value between 0.0 and 1.0
    DOT_THRESHOLD: Threshold for considering the two vectors as
                            colinear. Not recommended to alter this.
  Returns:
      Interpolation vector between v0 and v1
  '''
  assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

  # Normalize the vectors to get the directions and angles
  v0_norm = torch.norm(v0, dim=-1)
  v1_norm = torch.norm(v1, dim=-1)

  v0_normed = v0 / v0_norm.unsqueeze(-1)
  v1_normed = v1 / v1_norm.unsqueeze(-1)

  # Dot product with the normalized vectors
  dot = (v0_normed * v1_normed).sum(-1)
  dot_mag = dot.abs()

  # if dp is NaN, it's because the v0 or v1 row was filled with 0s
  # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
  gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
  can_slerp = ~gotta_lerp

  t_batch_dim_count = max(0, t.dim()-v0.dim()) if isinstance(t, torch.Tensor) else 0
  t_batch_dims = t.shape[:t_batch_dim_count] if isinstance(t, torch.Tensor) else torch.Size([])
  out = torch.zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

  # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
  if gotta_lerp.any():
    lerped = torch.lerp(v0, v1, t)

    out = lerped.where(gotta_lerp.unsqueeze(-1), out)

  # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
  if can_slerp.any():

    # Calculate initial angle between v0 and v1
    theta_0 = dot.arccos().unsqueeze(-1)
    sin_theta_0 = theta_0.sin()
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = theta_t.sin()
    # Finish the slerp algorithm
    s0 = (theta_0 - theta_t).sin() / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    slerped = s0 * v0 + s1 * v1

    out = slerped.where(can_slerp.unsqueeze(-1), out)
  
  return out

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def run_example(img_file, ref_img_file):
    return generate_image(img_file, ref_img_file, 25, 3, 23, 2, False)

def run_emoca(img, ref_img):

    img_dict = preprocess_for_emoca(img, face_detector)
    img_dict['image'] = img_dict['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        codedict = emoca_model.encode(img_dict, training=False)
    
    bbox, bbox_type, landmarks = face_detector.run(np.array(ref_img.convert('RGB')), with_landmarks=True)
    if len(bbox) == 0:
        raise gr.Error(f"Face detection failed in reference image! Please try with another reference image.")
    if len(bbox)>1:   # select largest face
        sizes = [(b[2]-b[0])*(b[3]-b[1]) for b in bbox]
        idx = np.argmax(sizes)
        lmks = landmarks[idx]
    else:
        lmks = landmarks[0]
    ref_img_aligned = image_align(ref_img.copy(), lmks, output_size=512)
    ref_img_dict = preprocess_for_emoca(ref_img_aligned, face_detector)
    ref_img_dict['image'] = ref_img_dict['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        ref_codedict = emoca_model.encode(ref_img_dict, training=False)
        ref_codedict['shapecode'] = codedict['shapecode'].clone()
        ref_codedict['detailcode'] = codedict['detailcode'].clone()
        tform = ref_img_dict['tform'].unsqueeze(0).to(device)
        tform = torch.inverse(tform).transpose(1, 2)
        visdict = emoca_model.decode(ref_codedict, training=False, render_orig=True, original_image=ref_img_dict['original_image'].unsqueeze(0).to(device), tform=tform)

    cond_img = Image.fromarray(((visdict['normal_images'][0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
    
    return ref_img_aligned, cond_img

def generate_image(criminal_path, accomplice_path, num_steps, guidance_scale, seed, num_images, use_lcm, interp_mode):

    if use_lcm:
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_lora()
    else:
        pipeline.disable_lora()
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    def open_img(f):
        from PIL import Image
        from PIL import ImageOps
        img = Image.open(f)
        img = ImageOps.exif_transpose(img)
        return img

    criminal_img = np.array(open_img(criminal_path))[:,:,::-1]
    accomplice_img = np.array(open_img(accomplice_path))[:,:,::-1]

    # Face detection and ID-embedding extraction
    print("Getting faces...")
    criminal_faces = app.get(criminal_img)
    accomplice_faces = app.get(accomplice_img)
    
    if len(criminal_faces) == 0:
        raise ValueError(f"Face detection on criminal failed! Please try with another input face image.")
    if len(accomplice_faces) == 0:
        raise ValueError(f"Face detection on accomplice failed! Please try with another input face image.")
    
    criminal_faces = sorted(criminal_faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
    criminal_id_emb = torch.tensor(criminal_faces['embedding'], dtype=dtype)[None].to(device)
    criminal_id_emb = criminal_id_emb/torch.norm(criminal_id_emb, dim=1, keepdim=True)   # normalize embedding
    accomplice_faces = sorted(accomplice_faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
    accomplice_id_emb = torch.tensor(accomplice_faces['embedding'], dtype=dtype)[None].to(device)
    accomplice_id_emb = accomplice_id_emb/torch.norm(accomplice_id_emb, dim=1, keepdim=True)   # normalize embedding

    print("Projeting face embedding...")
    if interp_mode in ("arcface-slerp", "arcface-lerp"):
        if interp_mode == "arcface-slerp":
            before_id_emb_interp = slerp(accomplice_id_emb, criminal_id_emb, 0.5)
        elif interp_mode == "arcface-lerp":
            before_id_emb_interp = 0.5 * accomplice_id_emb + 0.5 * criminal_id_emb
        before_id_emb_interp = before_id_emb_interp/torch.norm(before_id_emb_interp, dim=1, keepdim=True)  # normalize embedding
        id_emb_interp = project_face_embs(pipeline, before_id_emb_interp)    # pass throught the encoder
    else:  # encoded-slerp or encoded-lerp
        criminal_id_emb = project_face_embs(pipeline, criminal_id_emb)    # pass throught the encoder
        accomplice_id_emb = project_face_embs(pipeline, accomplice_id_emb)
        if interp_mode == "encoded-slerp":
            id_emb_interp = slerp(accomplice_id_emb, criminal_id_emb, 0.5)  # interpolate between the two embeddings
        else:  # encoded-lerp
            id_emb_interp = 0.5 * accomplice_id_emb + 0.5 * criminal_id_emb  # interpolate between the two embeddings

    # pose extraction with EMOCA
    print("Getting pose...")
    ref_img_a, cond_img = run_emoca(open_img(criminal_path), open_img(accomplice_path))
                    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("Start inference...")
    images = pipeline(
        image=cond_img,
        prompt_embeds=id_emb_interp,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale, 
        num_images_per_prompt=num_images,
        generator=generator
    ).images


    return images[0]

parser = argparse.ArgumentParser(description="Morphing pipeline")
parser.add_argument("--root", type=str, required=True, help="Root directory for the images")
parser.add_argument("--pairs", type=str, required=True, help="Path to the pairs file")
parser.add_argument("--output", type=str, required=True, help="Output directory for the generated images")
parser.add_argument("--num_steps", type=int, default=25, help="Number of diffusion steps")
parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale for the diffusion model")
parser.add_argument("--seed", type=int, default=23, help="Random seed for the diffusion model")
parser.add_argument("--use_lcm", action='store_true', help="Use LCM for latent consistency")
parser.add_argument("--randomize_seed", action='store_true', help="Randomize the seed for each image")
parser.add_argument("--interp_mode", choices=['arcface-slerp', 'arcface-lerp', 'encoded-lerp', 'encoded-slerp'], default='encoded-slerp', help="Interpolation mode for the morphing")
args = parser.parse_args()

with open(args.pairs, 'r') as f:
    pairs = f.readlines()
    pairs = [x.strip().split() for x in pairs]
    pairs = [(os.path.join(args.root, x[0]), os.path.join(args.root, x[1])) for x in pairs]
num_steps = args.num_steps
guidance_scale = args.guidance_scale
seed = args.seed
use_lcm = args.use_lcm
randomize_seed = args.randomize_seed
interp_mode = args.interp_mode
seed = randomize_seed_fn(seed, randomize_seed)

if not os.path.exists(args.output):
    os.makedirs(args.output)

for accomplice_path, criminal_path in pairs:
    criminal_path = criminal_path.strip()
    accomplice_path = accomplice_path.strip()
    accomplice_stem = accomplice_path.split("/")[-1].split(".")[0]
    criminal_stem = criminal_path.split("/")[-1].split(".")[0]
    fname = f"M_{accomplice_stem}_{criminal_stem}_C23_B50_W50_PA23_PM00_F00.png"
    image = generate_image(criminal_path, accomplice_path, num_steps, guidance_scale, seed, 1, use_lcm, interp_mode)
    image.save(f"{args.output}/{fname}")
