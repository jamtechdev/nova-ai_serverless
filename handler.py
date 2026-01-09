#!/usr/bin/env python3
"""
RunPod Serverless - NSFW Image Generator
Optimized prompts - Ethnicity FIRST within 77 token CLIP limit
"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import runpod
import torch
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import time
import base64
import re

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

# Check Compel support
COMPEL_AVAILABLE = False
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
    print("✅ Compel available")
except ImportError:
    print("⚠️ Compel not available")

# Optional Cloudinary
CLOUDINARY_AVAILABLE = False
try:
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name="dpofaoo6n",
        api_key="278875338772311",
        api_secret="CcghZeTAPMZYdly0SjxG69ENN2w"
    )
    CLOUDINARY_AVAILABLE = True
    print("✅ Cloudinary configured")
except:
    print("⚠️ Cloudinary not available")


MODEL_PATH = "/app/models/civitai_new.safetensors"

pipe = None
pipe_img2img = None
compel = None
compel_img2img = None


# ============================================
# OCCUPATION SETTINGS
# ============================================
OCCUPATION_SETTINGS = {
    "Custom": {"background": "studio", "lighting": "soft lighting"},
    "None": {"background": "bedroom", "lighting": "soft lighting"},
    "Life Coach": {"background": "office", "lighting": "warm light"},
    "Doctor": {"background": "medical office", "lighting": "bright"},
    "Teacher": {"background": "classroom", "lighting": "bright"},
    "Artist": {"background": "art studio", "lighting": "natural light"},
    "Chef": {"background": "kitchen", "lighting": "bright"},
    "Stripper": {"background": "club stage", "lighting": "neon lights"},
    "Dominatrix": {"background": "dungeon", "lighting": "red light"},
    "Lawyer": {"background": "office", "lighting": "office light"},
    "Engineer": {"background": "workshop", "lighting": "bright"},
    "Professional Gamer": {"background": "gaming room RGB", "lighting": "neon glow"},
    "Nurse": {"background": "hospital", "lighting": "clinical light"},
    "Secretary": {"background": "office", "lighting": "office light"},
    "Maid": {"background": "luxury bedroom", "lighting": "soft light"},
    "Fitness Trainer": {"background": "gym", "lighting": "bright"},
}


# ============================================
# QUALITY PRESETS - Optimized for Lustify V7
# Lustify recommends: CFG 2.5-4.5, Steps ~30, Highres 1.4-1.5x, denoise 0.4
# ============================================
QUALITY_PRESETS = {
    "standard": {"base_width": 896, "base_height": 1152, "steps": 25, "cfg": 3.5, "highres_scale": 1.4, "highres_steps": 20, "highres_denoise": 0.4},
    "hd": {"base_width": 896, "base_height": 1152, "steps": 30, "cfg": 3.5, "highres_scale": 1.5, "highres_steps": 25, "highres_denoise": 0.4},
    "ultra_hd": {"base_width": 896, "base_height": 1152, "steps": 35, "cfg": 3.5, "highres_scale": 1.5, "highres_steps": 30, "highres_denoise": 0.4},
    "extreme": {"base_width": 896, "base_height": 1152, "steps": 40, "cfg": 4.0, "highres_scale": 1.5, "highres_steps": 35, "highres_denoise": 0.45},
}


# ============================================
# NEGATIVE PROMPT - Lustify optimized (simple works better)
# ============================================
NEGATIVE_PROMPT = "worst quality, low quality, bad anatomy, bad hands, deformed, ugly, blurry, watermark, text, logo"
SFW_NEGATIVE_PROMPT = NEGATIVE_PROMPT + ", nude, naked, nsfw, explicit"


# ============================================
# OPTIMIZED PROMPTS FOR LUSTIFY V7
# - Understands danbooru tags + natural language
# - Simple prompts work better (no schizoprompting)
# - Uses camera tags, lighting, photography style
# - Ethnicity FIRST for CLIP attention
# ============================================
PROMPTS = {
    # === INTERCOURSE ===
    "doggy_style": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        doggystyle sex, 1girl 1boy, on all fours, ass up, pussy penetration from behind, moaning face,
        {setting_description}, glamour photography, bokeh, shot on Canon EOS 5D""",
    
    "missionary": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        missionary sex, 1girl 1boy, lying on back, legs spread, deep penetration, intimate,
        {setting_description}, cinematic lighting, shot on Canon EOS 5D""",
    
    "cowgirl": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description} bouncing,
        cowgirl riding, 1girl 1boy, woman on top, straddling, riding cock,
        {setting_description}, glamour photography, soft lighting""",
    
    "reverse_cowgirl": """{skin_description}, {age}yo woman, {hair_description}, {breast_description}, {butt_description},
        reverse cowgirl, 1girl 1boy, facing away, rear view, riding, ass visible,
        {setting_description}, cinematic lighting, bokeh""",
    
    "mating_press": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        mating press, 1girl 1boy, legs up, breeding position, deep penetration, ahegao,
        {setting_description}, dramatic lighting, shot on Canon EOS 5D""",
    
    "anal": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description}, {butt_description},
        anal sex, 1girl 1boy, from behind, ass penetration, 
        {setting_description}, cinematic lighting, glamour photography""",
    
    "pronebone": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        pronebone, 1girl 1boy, lying flat on stomach, pressed down, from behind,
        {setting_description}, soft lighting, intimate""",
    
    "against_wall": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        wall sex, 1girl 1boy, pressed against wall, legs wrapped, standing sex,
        {setting_description}, dramatic lighting, shot on Canon EOS 5D""",

    # === ORAL ===
    "blowjob": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        blowjob, 1girl 1boy, kneeling, cock in mouth, looking up, eye contact,
        {setting_description}, soft lighting, glamour photography""",
    
    "deepthroat": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        deepthroat, 1girl 1boy, throat penetration, tears, saliva, gagging,
        {setting_description}, dramatic lighting, shot on Canon EOS 5D""",
    
    "titfuck": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        titfuck paizuri, 1girl 1boy, cock between breasts, tongue out, looking up,
        {setting_description}, soft lighting, glamour photography""",
    
    "cunnilingus": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        cunnilingus, 1girl 1boy, man eating pussy, legs spread, moaning, pleasure,
        {setting_description}, soft lighting, intimate, bokeh""",
    
    "face_sitting": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description}, {butt_description},
        facesitting, 1girl 1boy, sitting on face, dominant, pussy on mouth,
        {setting_description}, cinematic lighting""",
    
    "69_position": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        69 position, 1girl 1boy, mutual oral, woman on top,
        {setting_description}, soft lighting, intimate""",

    # === SOLO/MASTURBATION ===
    "fingering_solo": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        masturbation fingering, 1girl solo, fingers in pussy, legs spread, moaning,
        {setting_description}, soft lighting, glamour photography, bokeh""",
    
    "dildo_solo": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        dildo masturbation, 1girl solo, sex toy, penetration, pleasure,
        {setting_description}, soft lighting, glamour photography""",
    
    "squirting": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        squirting orgasm, 1girl solo, pussy gushing, ahegao, intense climax,
        {setting_description}, dramatic lighting, shot on Canon EOS 5D""",
    
    "vibrator": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        vibrator, 1girl solo, magic wand on pussy, trembling, moaning,
        {setting_description}, soft lighting, glamour photography""",

    # === BODY FOCUS ===
    "boobs_close": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        breast focus, 1girl solo, close up breasts, nipples, areolas detailed,
        studio lighting, glamour photography, shot on Canon EOS 5D, bokeh""",
    
    "pussy_close": """{skin_description}, {age}yo woman,
        pussy closeup, 1girl solo, labia detailed, wet, spread,
        studio lighting, glamour photography, shot on Canon EOS 5D""",
    
    "ass_close": """{skin_description}, {age}yo woman, {hair_description}, {butt_description},
        ass closeup, 1girl solo, bent over, rear view, asshole visible,
        studio lighting, glamour photography, shot on Canon EOS 5D""",
    
    "all_fours_rear": """{skin_description}, {age}yo woman, {hair_description}, {breast_description}, {butt_description},
        all fours, 1girl solo, rear view, ass up, presenting, looking back,
        {setting_description}, soft lighting, glamour photography""",
    
    "spread_ass": """{skin_description}, {age}yo woman, {hair_description}, {butt_description},
        spreading ass, 1girl solo, hands pulling cheeks, asshole exposed,
        {setting_description}, glamour photography, shot on Canon EOS 5D""",
    
    "spread_pussy": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        spreading pussy, 1girl solo, fingers spreading labia, pink inside, wet,
        {setting_description}, soft lighting, glamour photography""",

    # === AFTERMATH ===
    "creampie": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        creampie, 1girl, cum dripping from pussy, satisfied, post-sex,
        {setting_description}, soft lighting, intimate, bokeh""",
    
    "cumshot": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        facial cumshot, 1girl 1boy, cum on face, tongue out, happy,
        {setting_description}, soft lighting, glamour photography""",
    
    "cumshot_face": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        facial cumshot, 1girl 1boy, cum on face, tongue out, happy,
        {setting_description}, soft lighting, glamour photography""",
    
    "cum_on_body": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        cum on body, 1girl 1boy, cum on breasts and stomach, glazed,
        {setting_description}, soft lighting, glamour photography""",
    
    "bukkake": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        bukkake, 1girl multiple boys, covered in cum, multiple cumshots,
        {setting_description}, dramatic lighting""",

    # === BDSM ===
    "handcuffs": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        handcuffs bondage, 1girl solo, hands bound, submissive, vulnerable,
        {setting_description}, dramatic lighting, cinematic""",
    
    "collar_leash": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        collar and leash, 1girl 1boy, pet play, kneeling, submissive,
        {setting_description}, dramatic lighting""",
    
    "blindfolded": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        blindfolded, 1girl solo, sensory play, anticipation, vulnerable,
        {setting_description}, dramatic lighting, cinematic""",

    # === GROUP ===
    "gangbang": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        gangbang, 1girl multiple boys, group sex, overwhelmed,
        {setting_description}, dramatic lighting""",
    
    "double_penetration": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        double penetration, 1girl 2boys, pussy and anal filled, sandwiched,
        {setting_description}, cinematic lighting""",
    
    "threesome": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        threesome MFM, 1girl 2boys, spitroast, cock in mouth and pussy,
        {setting_description}, cinematic lighting""",
    
    "lesbian": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        lesbian sex, 2girls, kissing, tribbing, intimate, sensual,
        {setting_description}, soft lighting, glamour photography, bokeh""",

    # === MISC ===
    "standing": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description}, {butt_description},
        standing nude, 1girl solo, full body, confident pose,
        {setting_description}, glamour photography, soft lighting, shot on Canon EOS 5D""",
    
    "spread_legs_sitting": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        sitting spread legs, 1girl solo, pussy visible, seductive, inviting,
        {setting_description}, soft lighting, glamour photography""",
    
    "bent_over_solo": """{skin_description}, {age}yo woman, {hair_description}, {breast_description}, {butt_description},
        bent over, 1girl solo, ass up, looking back, teasing,
        {setting_description}, soft lighting, glamour photography""",
    
    "showering": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        showering, 1girl solo, wet skin, water droplets, steam,
        bathroom, soft lighting, glamour photography""",
    
    "bath": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        bathing, 1girl solo, in bathtub, bubbles, relaxed, wet,
        bathroom, warm lighting, glamour photography""",
    
    "yoga_pose": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        nude yoga, 1girl solo, downward dog pose, flexible, athletic,
        yoga studio, natural lighting, glamour photography""",
    
    "jack_o_pose": """{skin_description}, {age}yo woman, {hair_description}, {breast_description}, {butt_description},
        jack-o pose, 1girl solo, extreme back arch, ass up, face down,
        {setting_description}, dramatic lighting, shot on Canon EOS 5D""",
}


# ============================================
# SFW PROMPTS - Lustify optimized
# ============================================
SFW_PROMPTS = {
    "standing": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        standing pose, 1girl solo, casual clothes, confident, full body,
        {setting_description}, glamour photography, soft lighting, shot on Canon EOS 5D""",
    
    "sitting": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        sitting pose, 1girl solo, casual attire, relaxed, elegant,
        {setting_description}, glamour photography, soft lighting, bokeh""",
    
    "portrait": """{skin_description}, {age}yo woman, {hair_description}, {eye_description},
        portrait headshot, 1girl solo, beautiful face, natural smile, elegant,
        studio portrait, glamour photography, soft lighting, shot on Canon EOS 5D, bokeh""",
}


# ============================================
# MODEL LOADING
# ============================================
def load_models():
    global pipe, pipe_img2img, compel, compel_img2img
    if pipe is not None:
        return pipe, pipe_img2img
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    print(f"\n🎨 Loading model: {MODEL_PATH}")
    
    pipe = StableDiffusionXLPipeline.from_single_file(MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
    pipe_img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae, text_encoder=pipe.text_encoder, text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer, tokenizer_2=pipe.tokenizer_2, unet=pipe.unet, scheduler=pipe.scheduler
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", solver_order=2
    )
    pipe_img2img.scheduler = pipe.scheduler
    
    pipe = pipe.to("cuda")
    pipe_img2img = pipe_img2img.to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe_img2img.enable_vae_slicing()
    pipe_img2img.enable_vae_tiling()
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        pipe_img2img.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    # Setup Compel for long prompt handling
    if COMPEL_AVAILABLE:
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        compel_img2img = Compel(
            tokenizer=[pipe_img2img.tokenizer, pipe_img2img.tokenizer_2],
            text_encoder=[pipe_img2img.text_encoder, pipe_img2img.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        print("✅ Compel configured for long prompts")
    
    print("✅ Model loaded!")
    return pipe, pipe_img2img


# ============================================
# HELPER FUNCTIONS
# ============================================
def safe_format(template: str, **kwargs) -> str:
    """Format string with fallback for missing keys"""
    placeholders = re.findall(r'\{(\w+)\}', template)
    for key in placeholders:
        if key not in kwargs:
            kwargs[key] = ""
    return template.format(**kwargs)


def enhance_image(image: Image.Image) -> Image.Image:
    image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
    for enhance_type, factor in [(ImageEnhance.Contrast, 1.05), (ImageEnhance.Color, 1.02), (ImageEnhance.Sharpness, 1.05)]:
        image = enhance_type(image).enhance(factor)
    return image


def upload_to_cloudinary(image: Image.Image, character_name: str, pose: str, seed: int) -> dict:
    if not CLOUDINARY_AVAILABLE:
        return None
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=85)
    buffer.seek(0)
    folder = f"nsfw_generations/{character_name.lower().replace(' ', '_')}"
    result = cloudinary.uploader.upload(buffer, public_id=f"{pose}_{seed}", folder=folder, format="webp", overwrite=True)
    return {"url": result['secure_url'], "public_id": result['public_id']}


def parse_character(character: dict) -> dict:
    """Parse character data from frontend payload"""
    
    # Hair
    hair_style = character.get("hairStyle", "long").lower()
    hair_colors = ['blonde', 'brunette', 'red', 'black', 'brown', 'white', 'pink', 'blue', 'silver', 'purple']
    hair_color = ""
    description = character.get("description", "").lower()
    for color in hair_colors:
        if color in description:
            hair_color = color
            break
    hair_description = f"{hair_color} {hair_style} hair".strip()
    
    # Eyes
    eye_color = character.get("eyeColor", "blue").lower()
    eye_description = f"{eye_color} eyes"
    
    # Breast
    breast_size = character.get("breastSize", "Large").lower()
    breast_map = {
        "small": "small breasts",
        "medium": "medium breasts", 
        "large": "large breasts",
        "extra-large": "huge breasts"
    }
    breast_description = breast_map.get(breast_size, "large breasts")
    
    # Butt
    butt_size = character.get("buttSize", "Medium").lower()
    butt_map = {
        "small": "small butt",
        "medium": "round butt",
        "large": "big butt bubble butt",
        "extra-large": "huge butt thicc"
    }
    butt_description = butt_map.get(butt_size, "round butt")
    
    # === ETHNICITY - CRITICAL FOR CLIP FIRST 77 TOKENS ===
    ethnicity = character.get("ethnicity", "white").lower()
    
    # Short but effective ethnicity descriptions
    ethnicity_map = {
        # Real ethnicities
        "asian": "asian woman, east asian, porcelain skin, asian face, monolid",
        "black": "black woman, ebony skin, african features, dark skin beauty",
        "white": "caucasian woman, fair skin, european features",
        "latina": "latina woman, tan skin, hispanic features, curvy",
        "arab": "arab woman, middle eastern, olive skin, exotic beauty",
        "indian": "indian woman, south asian, brown skin, desi beauty",
        
        # Fantasy ethnicities
        "elf": "elf woman, pointed elf ears, ethereal beauty, elven features, fantasy elf",
        "alien": "alien woman, otherworldly, pointed ears, exotic alien, sci-fi",
        "demon": "demon woman, succubus, small horns, demonic beauty, supernatural",
    }
    
    skin_description = ethnicity_map.get(ethnicity, ethnicity_map["white"])
    
    # Occupation/setting
    personality = character.get("personalityId", {}) or {}
    occupation = personality.get("occupationId", "None")
    occ_setting = OCCUPATION_SETTINGS.get(occupation, OCCUPATION_SETTINGS["None"])
    setting_description = f"{occ_setting['background']}, {occ_setting['lighting']}"
    
    return {
        "age": character.get("age", 25),
        "name": character.get("name", "unknown"),
        "ethnicity": ethnicity,
        "hair_description": hair_description,
        "eye_description": eye_description,
        "breast_description": breast_description,
        "butt_description": butt_description,
        "skin_description": skin_description,
        "setting_description": setting_description,
        "occupation": occupation
    }


# ============================================
# MAIN HANDLER
# ============================================
def handler(event):
    """Handle image generation - accepts frontend payload format"""
    try:
        input_data = event.get("input", {})
        
        # Parse character
        character = input_data.get("character", {})
        if character:
            features = parse_character(character)
        else:
            features = {
                "age": input_data.get("age", 25),
                "name": input_data.get("character_name", "unknown"),
                "ethnicity": input_data.get("ethnicity", "white"),
                "hair_description": input_data.get("hair_description", "long hair"),
                "eye_description": input_data.get("eye_description", "blue eyes"),
                "breast_description": input_data.get("breast_description", "large breasts"),
                "butt_description": input_data.get("butt_description", "round butt"),
                "skin_description": input_data.get("skin_description", "caucasian woman, fair skin"),
                "setting_description": input_data.get("setting_description", "bedroom, soft lighting"),
            }
        
        # Get pose
        pose_name = input_data.get("pose_name", "")
        if not pose_name and character:
            personality = character.get("personalityId", {}) or {}
            pose_name = personality.get("poseId", "standing")
        if not pose_name:
            pose_name = "standing"
        
        pose_name = pose_name.lower().replace(' ', '_').replace('-', '_')
        
        # Other params
        quality = input_data.get("quality", "hd")
        seed = input_data.get("seed") or -1
        use_highres = input_data.get("use_highres", True)
        enhance_img = input_data.get("enhance", True)
        nsfw = input_data.get("nsfw", True)
        upload_cloud = input_data.get("upload_cloudinary", True)
        
        # Select prompts
        if nsfw:
            prompts_dict = PROMPTS
            negative_prompt = NEGATIVE_PROMPT
        else:
            prompts_dict = SFW_PROMPTS
            negative_prompt = SFW_NEGATIVE_PROMPT
            if pose_name not in prompts_dict:
                pose_name = "standing"
        
        # Build prompt
        if pose_name in prompts_dict:
            prompt = safe_format(prompts_dict[pose_name], **features)
        else:
            prompt = safe_format(prompts_dict.get("standing", PROMPTS["standing"]), **features)
            pose_name = "standing"
        
        # Clean prompt - remove extra whitespace
        prompt = ' '.join(prompt.split())
        
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["hd"])
        
        if seed == -1 or seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"🎨 Generating: {features['name']}")
        print(f"   Ethnicity: {features['ethnicity']}")
        print(f"   Pose: {pose_name} | Quality: {quality}")
        print(f"   Prompt (first 200 chars): {prompt[:200]}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load model
        model, model_img2img = load_models()
        
        # Generate with Compel if available (handles long prompts better)
        if COMPEL_AVAILABLE and compel:
            print("📝 Using Compel for prompt encoding")
            conditioning, pooled = compel(prompt)
            neg_conditioning, neg_pooled = compel(negative_prompt)
            
            image = model(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                width=preset['base_width'],
                height=preset['base_height'],
                num_inference_steps=preset['steps'],
                guidance_scale=preset['cfg'],
                generator=generator,
                clip_skip=2
            ).images[0]
        else:
            print("📝 Using standard prompt encoding")
            image = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=preset['base_width'],
                height=preset['base_height'],
                num_inference_steps=preset['steps'],
                guidance_scale=preset['cfg'],
                generator=generator,
                clip_skip=2
            ).images[0]
        
        # Highres
        if use_highres:
            final_w = int(preset['base_width'] * preset['highres_scale'])
            final_h = int(preset['base_height'] * preset['highres_scale'])
            upscaled = image.resize((final_w, final_h), Image.LANCZOS)
            generator = torch.Generator(device="cuda").manual_seed(seed + 1)
            
            if COMPEL_AVAILABLE and compel_img2img:
                conditioning, pooled = compel_img2img(prompt)
                neg_conditioning, neg_pooled = compel_img2img(negative_prompt)
                
                image = model_img2img(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=neg_conditioning,
                    negative_pooled_prompt_embeds=neg_pooled,
                    image=upscaled,
                    strength=preset['highres_denoise'],
                    num_inference_steps=preset['highres_steps'],
                    guidance_scale=preset['cfg'],
                    generator=generator
                ).images[0]
            else:
                image = model_img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=upscaled,
                    strength=preset['highres_denoise'],
                    num_inference_steps=preset['highres_steps'],
                    guidance_scale=preset['cfg'],
                    generator=generator
                ).images[0]
        else:
            final_w, final_h = preset['base_width'], preset['base_height']
        
        if enhance_img:
            image = enhance_image(image)
        
        gen_time = time.time() - start_time
        print(f"✅ Image: {final_w}x{final_h} in {gen_time:.1f}s")
        
        # Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        result = {
            "success": True,
            "image": img_base64,
            "seed": seed,
            "character_name": features["name"],
            "ethnicity": features["ethnicity"],
            "pose": pose_name,
            "occupation": features.get("occupation", "None"),
            "quality": quality,
            "resolution": f"{final_w}x{final_h}",
            "generation_time": f"{gen_time:.2f}s"
        }
        
        # Cloudinary
        if upload_cloud and CLOUDINARY_AVAILABLE:
            cloud_result = upload_to_cloudinary(image, features["name"], pose_name, seed)
            if cloud_result:
                result["image_url"] = cloud_result["url"]
                result["cloudinary_url"] = cloud_result["url"]
                result["cloudinary_public_id"] = cloud_result["public_id"]
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


runpod.serverless.start({"handler": handler})
