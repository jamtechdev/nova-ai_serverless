#!/usr/bin/env python3
"""
RunPod Serverless - NSFW Image Generator
Model downloaded at build time - No Network Volume needed
"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import runpod
import torch
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageEnhance, ImageFilter
import time
import base64

# Check Compel support
COMPEL_AVAILABLE = False
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
    print("✅ Compel available for long prompts")
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

# ============================================
# MODEL PATH - Built into Docker image
# ============================================
MODEL_PATH = "/app/models/civitai_new.safetensors"

# Check if model exists
if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024 * 1024)
    print(f"✅ Model found: {MODEL_PATH} ({file_size:.2f} GB)")
else:
    print(f"❌ Model NOT found at: {MODEL_PATH}")
    # List what's in /app/models
    if os.path.exists("/app/models"):
        print(f"📁 Contents of /app/models: {os.listdir('/app/models')}")
    else:
        print("📁 /app/models directory does not exist!")

# ============================================
# QUALITY PRESETS
# ============================================
QUALITY_PRESETS = {
    "standard": {
        "base_width": 896, "base_height": 1152, "steps": 35, "cfg": 5.0,
        "highres_scale": 1.5, "highres_steps": 25, "highres_denoise": 0.4,
    },
    "hd": {
        "base_width": 896, "base_height": 1152, "steps": 40, "cfg": 5.0,
        "highres_scale": 1.5, "highres_steps": 30, "highres_denoise": 0.45,
    },
    "ultra_hd": {
        "base_width": 896, "base_height": 1152, "steps": 50, "cfg": 5.0,
        "highres_scale": 1.5, "highres_steps": 35, "highres_denoise": 0.45,
    },
    "extreme": {
        "base_width": 896, "base_height": 1152, "steps": 60, "cfg": 5.0,
        "highres_scale": 1.5, "highres_steps": 40, "highres_denoise": 0.5,
    }
}

# ============================================
# GLOBAL MODELS
# ============================================
pipe = None
pipe_img2img = None
compel = None
compel_img2img = None

def load_models():
    global pipe, pipe_img2img, compel, compel_img2img
    if pipe is not None:
        return pipe, pipe_img2img
    
    # Verify model exists before loading
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please check the build logs.")
    
    file_size = os.path.getsize(MODEL_PATH)
    if file_size < 1000000:  # Less than 1MB means download failed
        raise ValueError(f"Model file is too small ({file_size} bytes). Download may have failed.")
    
    print(f"\n🔥 Loading model: {MODEL_PATH}")
    print(f"   File size: {file_size / (1024*1024*1024):.2f} GB")
    
    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    )
    
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
        print("✅ xformers enabled")
    except:
        print("⚠️ xformers not available")
    
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
        print("✅ Compel initialized!")
    
    print("✅ Model loaded!\n")
    return pipe, pipe_img2img


# ============================================
# LONG PROMPT HANDLING
# ============================================
def split_prompt_for_sdxl(prompt: str) -> tuple:
    parts = [p.strip() for p in prompt.split(',') if p.strip()]
    
    if len(parts) <= 30:
        return (prompt, prompt)
    
    style_keywords = [
        'photorealistic', 'realistic', '8k', 'uhd', 'masterpiece', 'detailed',
        'lighting', 'cinematic', 'dramatic', 'shadows', 'soft', 'warm', 'golden',
        'professional', 'photography', 'raw', 'quality', 'resolution', 'sharp',
    ]
    
    prompt_1_parts = []
    prompt_2_parts = []
    
    for part in parts:
        part_lower = part.lower()
        if any(kw in part_lower for kw in style_keywords):
            prompt_2_parts.append(part)
        else:
            prompt_1_parts.append(part)
    
    max_parts = 35
    if len(prompt_1_parts) > max_parts:
        overflow = prompt_1_parts[max_parts:]
        prompt_1_parts = prompt_1_parts[:max_parts]
        prompt_2_parts = overflow + prompt_2_parts
    
    if len(prompt_2_parts) > max_parts:
        prompt_2_parts = prompt_2_parts[:max_parts]
    
    return (', '.join(prompt_1_parts), ', '.join(prompt_2_parts))


# ============================================
# NEGATIVE PROMPT
# ============================================
NEGATIVE_PROMPT = """score_6, score_5, score_4, (worst quality:1.4), (low quality:1.4),
bad anatomy, bad hands, extra fingers, missing fingers, ugly, deformed,
anime, cartoon, drawing, 3d render, cgi, plastic skin, blurry, watermark, text"""


# ============================================
# ALL PROMPTS
# ============================================
PROMPTS = {
    # INTERCOURSE CATEGORY
    "doggy_style": """score_9, score_8_up, score_7_up, 1girl 1boy, doggystyle sex from behind, on all fours,
        woman on hands and knees on bed, deeply arched back, ass raised high, head down moaning,
        {age} years old, {hair_description}, {eye_description}, gorgeous face showing pleasure,
        {breast_description} hanging swaying, {skin_description}, athletic body, round bubble butt jiggling,
        pussy visible from behind stretched around cock, deep penetration, man kneeling behind gripping hips,
        thick penis penetrating deeply, intense thrusting, ass rippling from impact,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, masterpiece""",
    
    "missionary": """score_9, score_8_up, score_7_up, 1girl 1boy, missionary position sex, face to face,
        woman lying on back on bed, legs spread wide, hair spread on pillow,
        {age} years old, {hair_description}, {eye_description}, face looking up with eye contact,
        {breast_description} visible, {skin_description}, narrow waist, wide hips,
        pussy stretched around penetrating penis, deep penetration visible,
        athletic man on top between legs, arms supporting, thick penis penetrating,
        legs wrapped around waist, passionate intimate sex,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, masterpiece""",
    
    "cowgirl": """score_9, score_8_up, score_7_up, 1girl 1boy, cowgirl riding position, woman on top,
        woman straddling riding, sitting on hips, knees on bed, thighs flexed,
        {age} years old, {hair_description} flowing with motion, {eye_description}, face tilted back moaning,
        {breast_description} bouncing wildly, {skin_description}, athletic body, round bubble butt,
        pussy stretched around cock, riding motion, deep penetration,
        man lying underneath, hands gripping her hips, powerful bouncing motion,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, masterpiece""",
    
    "reverse_cowgirl": """score_9, score_8_up, score_7_up, 1girl 1boy, reverse cowgirl, woman facing away, rear view,
        woman straddling facing away, back visible, knees on bed, athletic legs,
        {age} years old, {hair_description} cascading down back, looking over shoulder,
        athletic body from behind, narrow waist, round ass spread, {skin_description},
        pussy stretched around cock from behind, asshole visible above,
        man lying on back, hands on her ass spreading, bouncing on cock, ass prominent,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, masterpiece""",
    
    "mating_press": """score_9, score_8_up, score_7_up, 1girl 1boy, mating press breeding position, legs pushed back,
        woman lying on back, legs folded against chest, knees near shoulders,
        {age} years old, {hair_description} messy wild, eyes rolled back, mouth gaped, ahegao,
        {breast_description} squeezed between thighs, {skin_description} flushed sweating,
        pussy fully exposed tilted up, maximum penetration, labia stretched,
        man on top pressing down, thick penis buried completely, intense pounding, breeding,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, intense, masterpiece""",

    "anal": """score_9, score_8_up, score_7_up, 1girl 1boy, anal sex penetration, from behind, ass prominent,
        woman on hands and knees, ass raised high, back arched, ass cheeks spread,
        {age} years old, {hair_description} loose, {eye_description} showing intensity,
        {breast_description} hanging, {skin_description} glistening,
        hands spreading ass cheeks, asshole stretched around thick cock, anal penetration,
        pussy visible below, lube glistening, man behind, controlled deep anal thrusting,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, anal detailed, masterpiece""",

    # ORAL CATEGORY
    "blowjob": """score_9, score_8_up, score_7_up, 1girl 1boy, blowjob oral sex, cock in mouth, kneeling,
        woman kneeling, face at crotch level, mouth wrapped around cock,
        {age} years old, {hair_description}, {eye_description} looking up eye contact,
        {breast_description} hanging, {skin_description},
        mouth stretched around thick penis, cheeks hollowing, saliva dripping,
        hands on thighs or stroking shaft, tongue visible, man standing hands on head,
        rhythmic head bobbing, intense oral pleasure,
        {setting_description}, photorealistic, 8K UHD, oral detailed, masterpiece""",
    
    "deepthroat": """score_9, score_8_up, score_7_up, 1girl 1boy, deepthroat oral sex, throat penetration,
        woman kneeling or head off bed edge, throat aligned, mouth wide,
        {age} years old, {hair_description}, eyes watering tears streaming, mascara running,
        {breast_description} visible, {skin_description} flushed,
        entire cock buried in throat, nose against pubic area, throat bulging,
        saliva flooding, drool pouring, gagging, man controlling head,
        {setting_description}, photorealistic, 8K UHD, extreme oral, masterpiece""",

    "titfuck": """score_9, score_8_up, score_7_up, 1girl 1boy, titfuck paizuri, penis between breasts,
        woman kneeling, large breasts squeezed together around cock,
        {age} years old, {hair_description}, {eye_description} seductive, tongue licking tip,
        extremely {breast_description} squeezing shaft, nipples erect, cleavage deep,
        {skin_description}, breasts glistening with oil,
        hands pressing breasts together, cock sliding in cleavage, tip emerging near face,
        {setting_description}, photorealistic, 8K UHD, titfuck detailed, masterpiece""",

    "cunnilingus": """score_9, score_8_up, score_7_up, 1girl 1boy, cunnilingus eating pussy, oral on woman,
        woman lying on back, legs spread wide, man's head between thighs,
        {age} years old, {hair_description} spread on pillow, {eye_description} rolled back, moaning,
        {breast_description} rising falling, {skin_description} flushed,
        legs spread maximally, pussy exposed, man's face buried in pussy, tongue licking,
        hands gripping his hair, thighs trembling, building to orgasm,
        {setting_description}, photorealistic, 8K UHD, oral pleasure, masterpiece""",

    # MASTURBATION & SOLO
    "fingering_solo": """score_9, score_8_up, score_7_up, 1girl solo, masturbation fingering, self pleasure,
        woman lying on back, legs spread wide, fingers between legs,
        {age} years old, {hair_description}, {eye_description} half-closed, moaning,
        {breast_description} one hand squeezing, {skin_description} flushed,
        legs spread, pussy exposed, fingers inside vagina, thumb on clit,
        fingers moving rhythmically, deep fingering, building to climax,
        {setting_description}, photorealistic, 8K UHD, solo intimate, masterpiece""",
    
    "dildo_solo": """score_9, score_8_up, score_7_up, 1girl solo, dildo sex toy, masturbation with toy,
        woman lying back, legs spread, using dildo on herself,
        {age} years old, {hair_description} messy, {eye_description} showing pleasure,
        {breast_description}, {skin_description} sweating,
        legs spread, large dildo penetrating pussy, toy visible clearly,
        hand moving dildo in and out, intense toy masturbation, building orgasm,
        {setting_description}, photorealistic, 8K UHD, toy detailed, masterpiece""",

    "squirting": """score_9, score_8_up, score_7_up, 1girl, squirting female ejaculation, intense orgasm,
        woman lying back, legs spread maximally, body convulsing,
        {age} years old, {hair_description} wild, eyes rolled back, mouth gaped, ahegao,
        {breast_description}, {skin_description} flushed red sweating,
        legs spread shaking, pussy gushing liquid, squirting stream visible,
        fingers stimulating rapidly, body convulsing, overwhelming climax,
        {setting_description}, sheets soaked, photorealistic, 8K UHD, squirting visible, masterpiece""",

    # BODY FOCUS
    "boobs_close": """score_9, score_8_up, score_7_up, 1girl solo, close-up breasts, breast focus,
        woman topless, upper body focus, close framing on breasts,
        {age} years old, {hair_description} partially visible,
        extremely {breast_description} close-up filling frame, nipples prominent, areolas detailed,
        {skin_description} realistic texture with pores,
        breasts natural hang, cleavage deep, soft breast texture,
        dramatic lighting on breasts, shadows enhancing curves,
        studio, photorealistic, 8K UHD, extreme breast detail, masterpiece""",
    
    "pussy_close": """score_9, score_8_up, score_7_up, 1girl solo, close-up pussy, genital focus, explicit macro,
        woman lying back, legs spread wide, genital area focus,
        {age} years old, body partially visible, {skin_description},
        pussy centered filling frame, labia detailed, inner labia pink,
        vaginal opening visible, clit prominent, wetness glistening,
        extreme genital detail, realistic anatomy,
        dramatic focused lighting, photorealistic, 8K UHD, explicit detailed, masterpiece""",
    
    "ass_close": """score_9, score_8_up, score_7_up, 1girl solo, close-up ass, butt focus, rear view,
        woman bent over presenting ass, ass filling frame,
        {age} years old, {hair_description} visible, {skin_description},
        round bubble butt centered, ass cheeks full firm, detailed texture,
        asshole visible between cheeks, pussy visible below,
        dramatic lighting on curves, shadows enhancing roundness,
        studio, photorealistic, 8K UHD, extreme ass detail, masterpiece""",

    "all_fours_rear": """score_9, score_8_up, score_7_up, 1girl solo, all fours position, rear view, ass up,
        woman on hands and knees, back arched, ass raised high,
        {age} years old, {hair_description}, looking back over shoulder seductive,
        {breast_description} hanging visible from side, {skin_description},
        on all fours, ass raised prominent, pussy visible from behind, asshole visible,
        presenting position, inviting approach,
        {setting_description}, photorealistic, 8K UHD, rear view detailed, masterpiece""",

    # AFTERMATH
    "creampie": """score_9, score_8_up, score_7_up, 1girl, creampie cum dripping from pussy, aftermath,
        woman lying back, legs spread, relaxed post-sex, pussy exposed,
        {age} years old, {hair_description} messy, {eye_description} half-closed tired, satisfied,
        {breast_description}, {skin_description} glistening sweat,
        legs spread, pussy red used, thick white cum dripping out,
        semen flowing from vagina, creampie leaking, cum pooling, post-orgasm bliss,
        {setting_description}, photorealistic, 8K UHD, creampie detailed, masterpiece""",
    
    "cumshot_face": """score_9, score_8_up, score_7_up, 1girl 1boy, facial cumshot, cum on face,
        woman kneeling, face tilted up, cum covering face,
        {age} years old, {hair_description} may have cum, {eye_description} looking up,
        {breast_description} may have cum, {skin_description} with white cum,
        face covered with cum, semen on cheeks nose forehead chin,
        thick ropes of cum, dripping, tongue out catching, mouth open,
        {setting_description}, photorealistic, 8K UHD, facial detailed, masterpiece""",

    # MISC
    "standing": """score_9, score_8_up, score_7_up, 1girl solo, standing nude, confident pose,
        woman standing upright, nude, legs slightly apart, confident,
        {age} years old, {hair_description}, {eye_description} looking at camera,
        {breast_description} natural position, {skin_description},
        standing straight confident, full nude body displayed,
        pussy visible between legs, comfortable nude,
        {setting_description}, photorealistic, 8K UHD, full body, masterpiece""",
    
    "spread_legs_sitting": """score_9, score_8_up, score_7_up, 1girl solo, sitting spread legs, seductive,
        woman sitting, legs spread wide, exposed, inviting,
        {age} years old, {hair_description}, {eye_description} seductive beckoning,
        {breast_description}, {skin_description},
        sitting with legs spread, pussy fully visible exposed,
        inviting spread, displaying while seated,
        {setting_description}, photorealistic, 8K UHD, spread detailed, masterpiece""",
    
    "bent_over_solo": """score_9, score_8_up, score_7_up, 1girl solo, bent over, ass up, tease,
        woman standing bent over forward, ass raised high, looking back,
        {age} years old, {hair_description}, {eye_description} teasing playful,
        {breast_description} hanging visible from side, {skin_description},
        bent over, ass raised prominent, pussy visible from behind,
        teasing bent over pose, presenting rear,
        {setting_description}, photorealistic, 8K UHD, rear view, masterpiece""",

    "showering_solo": """score_9, score_8_up, score_7_up, 1girl solo, showering wet, water running,
        woman standing in shower, water cascading over body, completely wet,
        {age} years old, {hair_description} completely wet slicked, {eye_description},
        {breast_description} wet water streaming, {skin_description} glistening wet,
        standing in shower, water running over body, steam surrounding,
        arms raised washing hair, water droplets everywhere,
        shower setting, photorealistic, 8K UHD, wet skin detailed, masterpiece""",
}


# ============================================
# POST-PROCESSING
# ============================================
def enhance_image(image: Image.Image) -> Image.Image:
    image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.05)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.02)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.05)
    return image


# ============================================
# CLOUDINARY UPLOAD
# ============================================
def upload_to_cloudinary(image: Image.Image, character_name: str, pose: str, seed: int) -> dict:
    if not CLOUDINARY_AVAILABLE:
        return None
    
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=85)
    buffer.seek(0)
    
    char_folder = character_name.lower().replace(' ', '_')
    filename = f"{pose}_{seed}"
    folder = f"nsfw_generations/{char_folder}"
    
    upload_result = cloudinary.uploader.upload(
        buffer, public_id=filename, folder=folder, 
        resource_type="image", format="webp", overwrite=True
    )
    
    return {
        "url": upload_result['secure_url'],
        "public_id": upload_result['public_id']
    }


# ============================================
# MAIN HANDLER
# ============================================
def handler(event):
    try:
        input_data = event.get("input", {})
        
        # Extract parameters
        prompt = input_data.get("prompt", "")
        negative_prompt = input_data.get("negative_prompt", NEGATIVE_PROMPT)
        pose_name = input_data.get("pose", "standing")
        quality = input_data.get("quality", "ultra_hd")
        seed = input_data.get("seed", -1)
        use_highres = input_data.get("use_highres", True)
        enhance = input_data.get("enhance", True)
        character_name = input_data.get("character_name", "unknown")
        upload_cloudinary = input_data.get("upload_cloudinary", True)
        
        # Character features for template
        age = input_data.get("age", 25)
        hair_description = input_data.get("hair_description", "long blonde hair")
        eye_description = input_data.get("eye_description", "blue eyes")
        breast_description = input_data.get("breast_description", "large breasts")
        skin_description = input_data.get("skin_description", "fair skin")
        setting_description = input_data.get("setting_description", "bedroom, soft lighting")
        
        # Build prompt from template if pose provided
        if not prompt and pose_name in PROMPTS:
            prompt = PROMPTS[pose_name].format(
                age=age,
                hair_description=hair_description,
                eye_description=eye_description,
                breast_description=breast_description,
                skin_description=skin_description,
                setting_description=setting_description
            )
        elif not prompt:
            prompt = f"score_9, score_8_up, 1girl, {hair_description}, {eye_description}, {breast_description}, {skin_description}, {setting_description}, photorealistic, 8K UHD, masterpiece"
        
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["ultra_hd"])
        
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"\n{'='*50}")
        print(f"🎨 Generating: {pose_name}")
        print(f"   Quality: {quality}")
        print(f"   Seed: {seed}")
        print(f"{'='*50}")
        
        start = time.time()
        model, model_img2img = load_models()
        
        prompt_words = len(prompt.split())
        use_compel = COMPEL_AVAILABLE and prompt_words > 50
        
        if use_compel:
            print("📝 Using Compel...")
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
            ).images[0]
        else:
            print("📝 Using dual encoder...")
            prompt_1, prompt_2 = split_prompt_for_sdxl(prompt)
            neg_1, neg_2 = split_prompt_for_sdxl(negative_prompt)
            
            image = model(
                prompt=prompt_1,
                prompt_2=prompt_2,
                negative_prompt=neg_1,
                negative_prompt_2=neg_2,
                width=preset['base_width'],
                height=preset['base_height'],
                num_inference_steps=preset['steps'],
                guidance_scale=preset['cfg'],
                generator=generator,
                clip_skip=2,
            ).images[0]
        
        # Highres upscale
        if use_highres:
            print("🔍 Highres upscale...")
            final_w = int(preset['base_width'] * preset['highres_scale'])
            final_h = int(preset['base_height'] * preset['highres_scale'])
            
            upscaled = image.resize((final_w, final_h), Image.LANCZOS)
            generator = torch.Generator(device="cuda").manual_seed(seed + 1)
            
            if use_compel:
                image = model_img2img(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=neg_conditioning,
                    negative_pooled_prompt_embeds=neg_pooled,
                    image=upscaled,
                    strength=preset['highres_denoise'],
                    num_inference_steps=preset['highres_steps'],
                    guidance_scale=preset['cfg'],
                    generator=generator,
                ).images[0]
            else:
                image = model_img2img(
                    prompt=prompt_1,
                    prompt_2=prompt_2,
                    negative_prompt=neg_1,
                    negative_prompt_2=neg_2,
                    image=upscaled,
                    strength=preset['highres_denoise'],
                    num_inference_steps=preset['highres_steps'],
                    guidance_scale=preset['cfg'],
                    generator=generator,
                ).images[0]
        else:
            final_w, final_h = preset['base_width'], preset['base_height']
        
        # Enhance
        if enhance:
            image = enhance_image(image)
        
        gen_time = time.time() - start
        print(f"✅ Generated: {final_w}x{final_h} in {gen_time:.1f}s")
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        result = {
            "image": img_base64,
            "seed": seed,
            "pose": pose_name,
            "quality": quality,
            "resolution": f"{final_w}x{final_h}",
            "generation_time": f"{gen_time:.2f}s"
        }
        
        # Upload to Cloudinary if enabled
        if upload_cloudinary and CLOUDINARY_AVAILABLE:
            cloud_result = upload_to_cloudinary(image, character_name, pose_name, seed)
            if cloud_result:
                result["cloudinary_url"] = cloud_result["url"]
                result["cloudinary_public_id"] = cloud_result["public_id"]
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
