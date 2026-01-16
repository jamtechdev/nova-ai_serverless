#!/usr/bin/env python3
"""
RunPod Serverless - NSFW Image Generator
Dual Style Support: Realistic + Anime
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

# Supabase Storage
SUPABASE_AVAILABLE = False
supabase_client = None
try:
    from supabase import create_client
    
    # Configure your Supabase credentials here
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kyfhvltdlauacdtzpyeo.supabase.co")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt5Zmh2bHRkbGF1YWNkdHpweWVvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njc5NDk3MzcsImV4cCI6MjA4MzUyNTczN30.xpJEvbO6Prh2jUq7qR79yKpdl2_pNLbmv-ty3EZuUlw")
    SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "ai-images")
    
    if SUPABASE_URL and SUPABASE_KEY and "your-" not in SUPABASE_URL:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        SUPABASE_AVAILABLE = True
        print(f"✅ Supabase configured: {SUPABASE_URL}")
    else:
        print("⚠️ Supabase credentials not set")
except Exception as e:
    print(f"⚠️ Supabase not available: {e}")


# ============================================
# MODEL PATHS
# ============================================
REALISTIC_MODEL = "/app/models/realistic.safetensors"
ANIME_MODEL = "/app/models/anime.safetensors"

# Global model storage
models = {
    "realistic": {"pipe": None, "pipe_img2img": None, "compel": None, "compel_img2img": None},
    "anime": {"pipe": None, "pipe_img2img": None, "compel": None, "compel_img2img": None}
}


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
# QUALITY PRESETS - REALISTIC (Lustify)
# ============================================
REALISTIC_PRESETS = {
    "standard": {"base_width": 896, "base_height": 1152, "steps": 25, "cfg": 3.5, "highres_scale": 1.4, "highres_steps": 20, "highres_denoise": 0.4},
    "hd": {"base_width": 896, "base_height": 1152, "steps": 30, "cfg": 3.5, "highres_scale": 1.5, "highres_steps": 25, "highres_denoise": 0.4},
    "ultra_hd": {"base_width": 896, "base_height": 1152, "steps": 35, "cfg": 3.5, "highres_scale": 1.5, "highres_steps": 30, "highres_denoise": 0.4},
    "extreme": {"base_width": 896, "base_height": 1152, "steps": 40, "cfg": 4.0, "highres_scale": 1.5, "highres_steps": 35, "highres_denoise": 0.45},
}

# ============================================
# QUALITY PRESETS - ANIME
# ============================================
ANIME_PRESETS = {
    "standard": {"base_width": 832, "base_height": 1216, "steps": 25, "cfg": 7.0, "highres_scale": 1.5, "highres_steps": 15, "highres_denoise": 0.5},
    "hd": {"base_width": 832, "base_height": 1216, "steps": 30, "cfg": 7.0, "highres_scale": 1.5, "highres_steps": 20, "highres_denoise": 0.5},
    "ultra_hd": {"base_width": 832, "base_height": 1216, "steps": 35, "cfg": 7.0, "highres_scale": 1.5, "highres_steps": 25, "highres_denoise": 0.5},
    "extreme": {"base_width": 832, "base_height": 1216, "steps": 40, "cfg": 7.5, "highres_scale": 1.5, "highres_steps": 30, "highres_denoise": 0.55},
}


# ============================================
# NEGATIVE PROMPTS
# ============================================
REALISTIC_NEGATIVE = "worst quality, low quality, bad anatomy, bad hands, deformed, ugly, blurry, watermark, text, logo"
REALISTIC_SFW_NEGATIVE = REALISTIC_NEGATIVE + ", nude, naked, nsfw, explicit"

ANIME_NEGATIVE = "worst quality, low quality, bad anatomy, bad hands, extra fingers, missing fingers, deformed, ugly, blurry, watermark, text, 3d, realistic, photo"
ANIME_SFW_NEGATIVE = ANIME_NEGATIVE + ", nude, naked, nsfw, explicit"


# ============================================
# ETHNICITY - REALISTIC
# ============================================
REALISTIC_ETHNICITY = {
    "asian": "asian woman, east asian, porcelain skin, asian face, monolid",
    "black": "black woman, ebony skin, african features, dark skin beauty",
    "white": "caucasian woman, fair skin, european features",
    "latina": "latina woman, tan skin, hispanic features, curvy",
    "arab": "arab woman, middle eastern, olive skin, exotic beauty",
    "indian": "indian woman, south asian, brown skin, desi beauty",
    "elf": "elf woman, pointed elf ears, ethereal beauty, elven features, fantasy elf",
    "alien": "alien woman, otherworldly, pointed ears, exotic alien, sci-fi",
    "demon": "demon woman, succubus, small horns, demonic beauty, supernatural",
}

# ============================================
# ETHNICITY - ANIME
# ============================================
ANIME_ETHNICITY = {
    "asian": "1girl, pale skin",
    "black": "1girl, dark skin, dark-skinned female",
    "white": "1girl, pale skin",
    "latina": "1girl, tan, tanned",
    "arab": "1girl, tan, olive skin",
    "indian": "1girl, dark skin, brown skin",
    "elf": "1girl, elf, pointy ears, elf ears, fantasy",
    "alien": "1girl, alien, pointy ears, unusual skin color",
    "demon": "1girl, demon girl, horns, demon horns, succubus",
}


# ============================================
# REALISTIC PROMPTS (Lustify optimized)
# ============================================
REALISTIC_PROMPTS = {
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

    "handcuffs": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        handcuffs bondage, 1girl solo, hands bound, submissive, vulnerable,
        {setting_description}, dramatic lighting, cinematic""",
    
    "collar_leash": """{skin_description}, {age}yo woman, {hair_description}, {eye_description}, {breast_description},
        collar and leash, 1girl 1boy, pet play, kneeling, submissive,
        {setting_description}, dramatic lighting""",
    
    "blindfolded": """{skin_description}, {age}yo woman, {hair_description}, {breast_description},
        blindfolded, 1girl solo, sensory play, anticipation, vulnerable,
        {setting_description}, dramatic lighting, cinematic""",

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

REALISTIC_SFW_PROMPTS = {
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
# ANIME PROMPTS (Danbooru tags)
# ============================================
ANIME_PROMPTS = {
    "doggy_style": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        sex from behind, doggystyle, all fours, ass up, nude, vaginal, sweat, blush, open mouth,
        {setting_description}""",
    
    "missionary": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        missionary, lying, on back, legs spread, nude, vaginal, sex, blush, sweat,
        {setting_description}""",
    
    "cowgirl": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        cowgirl position, girl on top, straddling, bouncing breasts, nude, riding, blush, ahegao,
        {setting_description}""",
    
    "reverse_cowgirl": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description}, {butt_description},
        reverse cowgirl, ass, from behind, nude, riding, sweat,
        {setting_description}""",
    
    "mating_press": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        mating press, legs up, spread legs, nude, deep penetration, ahegao, rolling eyes, tongue out,
        {setting_description}""",
    
    "anal": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description}, {butt_description},
        anal, from behind, nude, ass, blush, sweat,
        {setting_description}""",
    
    "pronebone": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        prone bone, lying, on stomach, from behind, nude, sweat, blush,
        {setting_description}""",
    
    "against_wall": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        against wall, standing sex, leg up, nude, sweat, blush,
        {setting_description}""",

    "blowjob": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        fellatio, oral, kneeling, looking up, nude, saliva, blush,
        {setting_description}""",
    
    "deepthroat": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        deepthroat, irrumatio, tears, saliva, nude, blush,
        {setting_description}""",
    
    "titfuck": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        paizuri, breast press, looking up, tongue out, nude, blush,
        {setting_description}""",
    
    "cunnilingus": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        cunnilingus, oral, spread legs, pussy, nude, blush, moaning,
        {setting_description}""",
    
    "face_sitting": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description}, {butt_description},
        facesitting, sitting on face, ass, nude, blush,
        {setting_description}""",
    
    "69_position": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        69, mutual oral, nude, blush,
        {setting_description}""",

    "fingering_solo": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, masturbation, fingering, spread legs, nude, pussy, blush, sweat,
        {setting_description}""",
    
    "dildo_solo": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, dildo, sex toy, insertion, nude, blush,
        {setting_description}""",
    
    "squirting": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        solo, squirting, female ejaculation, ahegao, nude, blush, sweat,
        {setting_description}""",
    
    "vibrator": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, vibrator, sex toy, trembling, nude, blush, sweat,
        {setting_description}""",

    "boobs_close": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        solo, breast focus, nipples, areolae, nude, close-up,
        {setting_description}""",
    
    "pussy_close": """masterpiece, best quality, {skin_description},
        solo, pussy focus, spread pussy, nude, close-up,
        {setting_description}""",
    
    "ass_close": """masterpiece, best quality, {skin_description}, {hair_description}, {butt_description},
        solo, ass focus, from behind, anus, nude,
        {setting_description}""",
    
    "all_fours_rear": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description}, {butt_description},
        solo, all fours, ass up, from behind, looking back, nude,
        {setting_description}""",
    
    "spread_ass": """masterpiece, best quality, {skin_description}, {hair_description}, {butt_description},
        solo, spread ass, presenting, anus, nude,
        {setting_description}""",
    
    "spread_pussy": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, spread pussy, spread legs, pussy, nude,
        {setting_description}""",

    "creampie": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        creampie, cum in pussy, cum drip, after sex, nude, blush,
        {setting_description}""",
    
    "cumshot": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        facial, cum on face, tongue out, nude, blush, happy,
        {setting_description}""",
    
    "cumshot_face": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        facial, cum on face, tongue out, nude, blush, happy,
        {setting_description}""",
    
    "cum_on_body": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        cum on body, cum on breasts, cum on stomach, nude, blush,
        {setting_description}""",
    
    "bukkake": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        bukkake, multiple boys, cum everywhere, nude, blush,
        {setting_description}""",

    "handcuffs": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, handcuffs, bound wrists, bondage, nude, blush,
        {setting_description}""",
    
    "collar_leash": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        collar, leash, pet play, kneeling, nude, blush, submissive,
        {setting_description}""",
    
    "blindfolded": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        solo, blindfold, nude, blush,
        {setting_description}""",

    "gangbang": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        gangbang, multiple boys, group sex, nude, blush, sweat,
        {setting_description}""",
    
    "double_penetration": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        double penetration, 2boys, vaginal, anal, nude, blush, sweat, ahegao,
        {setting_description}""",
    
    "threesome": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        threesome, 2boys, spitroast, nude, blush,
        {setting_description}""",
    
    "lesbian": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        2girls, yuri, kiss, tribadism, nude, blush,
        {setting_description}""",

    "standing": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description}, {butt_description},
        solo, standing, nude, full body,
        {setting_description}""",
    
    "spread_legs_sitting": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, sitting, spread legs, pussy, nude,
        {setting_description}""",
    
    "bent_over_solo": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description}, {butt_description},
        solo, bent over, ass up, looking back, nude,
        {setting_description}""",
    
    "showering": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, shower, wet, water, nude, steam,
        {setting_description}""",
    
    "bath": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, bathing, onsen, water, wet, relaxed,
        {setting_description}""",
    
    "yoga_pose": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description},
        solo, yoga, flexible, nude, athletic,
        {setting_description}""",
    
    "jack_o_pose": """masterpiece, best quality, {skin_description}, {hair_description}, {breast_description}, {butt_description},
        solo, jack-o pose, ass up, face down, arched back, nude,
        {setting_description}""",
}

ANIME_SFW_PROMPTS = {
    "standing": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, standing, full body, casual clothes,
        {setting_description}""",
    
    "sitting": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description}, {breast_description},
        solo, sitting, relaxed, casual attire,
        {setting_description}""",
    
    "portrait": """masterpiece, best quality, {skin_description}, {hair_description}, {eye_description},
        solo, portrait, upper body, beautiful face, smile,
        {setting_description}""",
}


# ============================================
# MODEL LOADING
# ============================================
def load_model(style: str):
    """Load model based on style"""
    global models
    
    style = style.lower()
    if style not in ["realistic", "anime"]:
        style = "realistic"
    
    # Check if already loaded
    if models[style]["pipe"] is not None:
        return models[style]
    
    model_path = REALISTIC_MODEL if style == "realistic" else ANIME_MODEL
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\n🎨 Loading {style.upper()} model: {model_path}")
    
    pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16, use_safetensors=True)
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
    
    # Compel
    comp = None
    comp_img2img = None
    if COMPEL_AVAILABLE:
        comp = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        comp_img2img = Compel(
            tokenizer=[pipe_img2img.tokenizer, pipe_img2img.tokenizer_2],
            text_encoder=[pipe_img2img.text_encoder, pipe_img2img.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
    
    models[style] = {
        "pipe": pipe,
        "pipe_img2img": pipe_img2img,
        "compel": comp,
        "compel_img2img": comp_img2img
    }
    
    print(f"✅ {style.upper()} model loaded!")
    return models[style]


# ============================================
# HELPER FUNCTIONS
# ============================================
def safe_format(template: str, **kwargs) -> str:
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


def upload_to_supabase(image: Image.Image, character_name: str, pose: str, seed: int, style: str) -> dict:
    """Upload image to Supabase Storage and return public URL"""
    if not SUPABASE_AVAILABLE or not supabase_client:
        return None
    
    try:
        # Convert image to bytes
        buffer = BytesIO()
        image.save(buffer, format="WEBP", quality=85)
        image_bytes = buffer.getvalue()
        
        # Create file path: style/character_name/pose_seed_timestamp.webp
        timestamp = int(time.time())
        safe_name = character_name.lower().replace(' ', '_').replace('-', '_')
        file_path = f"{style}/{safe_name}/{pose}_{seed}_{timestamp}.webp"
        
        # Upload to Supabase Storage
        result = supabase_client.storage.from_(SUPABASE_BUCKET).upload(
            path=file_path,
            file=image_bytes,
            file_options={"content-type": "image/webp", "upsert": "true"}
        )
        
        # Get public URL
        public_url = supabase_client.storage.from_(SUPABASE_BUCKET).get_public_url(file_path)
        
        print(f"✅ Uploaded to Supabase: {file_path}")
        
        return {
            "url": public_url,
            "path": file_path,
            "bucket": SUPABASE_BUCKET
        }
        
    except Exception as e:
        print(f"❌ Supabase upload error: {e}")
        return None


def parse_character(character: dict, style: str) -> dict:
    """Parse character - different for each style"""
    
    # Hair
    hair_style = character.get("hairStyle", "long").lower()
    hair_colors = ['blonde', 'brunette', 'red', 'black', 'brown', 'white', 'pink', 'blue', 'silver', 'purple']
    hair_color = ""
    description = character.get("description", "").lower()
    for color in hair_colors:
        if color in description:
            hair_color = color
            break
    hair_description = f"{hair_color} {hair_style} hair".strip() if style == "realistic" else f"{hair_color} hair, {hair_style} hair"
    
    # Eyes
    eye_color = character.get("eyeColor", "blue").lower()
    eye_description = f"{eye_color} eyes"
    
    # Breast - different tags for anime
    breast_size = character.get("breastSize", "Large").lower()
    if style == "anime":
        breast_map = {
            "small": "small breasts, flat chest",
            "medium": "medium breasts",
            "large": "large breasts",
            "extra-large": "huge breasts, gigantic breasts"
        }
    else:
        breast_map = {
            "small": "small breasts",
            "medium": "medium breasts",
            "large": "large breasts",
            "extra-large": "huge breasts"
        }
    breast_description = breast_map.get(breast_size, "large breasts")
    
    # Butt
    butt_size = character.get("buttSize", "Medium").lower()
    if style == "anime":
        butt_map = {
            "small": "small ass",
            "medium": "ass",
            "large": "big ass, huge ass",
            "extra-large": "huge ass"
        }
    else:
        butt_map = {
            "small": "small butt",
            "medium": "round butt",
            "large": "big butt bubble butt",
            "extra-large": "huge butt thicc"
        }
    butt_description = butt_map.get(butt_size, "round butt")
    
    # Ethnicity
    ethnicity = character.get("ethnicity", "white").lower()
    ethnicity_map = ANIME_ETHNICITY if style == "anime" else REALISTIC_ETHNICITY
    skin_description = ethnicity_map.get(ethnicity, ethnicity_map.get("white", ""))
    
    # Setting
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
    """Handle image generation - Realistic or Anime"""
    try:
        input_data = event.get("input", {})
        
        # Get style
        character = input_data.get("character", {})
        style = character.get("style", input_data.get("style", "Realistic")).lower()
        if style not in ["realistic", "anime"]:
            style = "realistic"
        
        # Parse character
        if character:
            features = parse_character(character, style)
        else:
            features = {
                "age": input_data.get("age", 25),
                "name": input_data.get("character_name", "unknown"),
                "hair_description": input_data.get("hair_description", "long hair"),
                "eye_description": input_data.get("eye_description", "blue eyes"),
                "breast_description": input_data.get("breast_description", "large breasts"),
                "butt_description": input_data.get("butt_description", "round butt"),
                "skin_description": input_data.get("skin_description", ""),
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
        
        # Params
        quality = input_data.get("quality", "hd")
        seed = input_data.get("seed") or -1
        use_highres = input_data.get("use_highres", True)
        enhance_img = input_data.get("enhance", True)
        nsfw = input_data.get("nsfw", True)
        upload_cloud = input_data.get("upload_cloudinary", True)
        
        # Select prompts/presets based on style
        if style == "anime":
            prompts_dict = ANIME_PROMPTS if nsfw else ANIME_SFW_PROMPTS
            negative_prompt = ANIME_NEGATIVE if nsfw else ANIME_SFW_NEGATIVE
            presets = ANIME_PRESETS
        else:
            prompts_dict = REALISTIC_PROMPTS if nsfw else REALISTIC_SFW_PROMPTS
            negative_prompt = REALISTIC_NEGATIVE if nsfw else REALISTIC_SFW_NEGATIVE
            presets = REALISTIC_PRESETS
        
        # Fallback pose
        if pose_name not in prompts_dict:
            pose_name = "standing"
        
        # Build prompt
        prompt = safe_format(prompts_dict[pose_name], **features)
        prompt = ' '.join(prompt.split())
        
        preset = presets.get(quality, presets["hd"])
        
        if seed == -1 or seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"🎨 Generating [{style.upper()}]: {features['name']}")
        print(f"   Ethnicity: {features.get('ethnicity', 'N/A')}")
        print(f"   Pose: {pose_name} | Quality: {quality}")
        print(f"   Prompt: {prompt[:150]}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load model
        model_data = load_model(style)
        pipe = model_data["pipe"]
        pipe_img2img = model_data["pipe_img2img"]
        comp = model_data["compel"]
        comp_img2img = model_data["compel_img2img"]
        
        # Generate
        if COMPEL_AVAILABLE and comp:
            conditioning, pooled = comp(prompt)
            neg_conditioning, neg_pooled = comp(negative_prompt)
            
            image = pipe(
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
            image = pipe(
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
            
            if COMPEL_AVAILABLE and comp_img2img:
                conditioning, pooled = comp_img2img(prompt)
                neg_conditioning, neg_pooled = comp_img2img(negative_prompt)
                
                image = pipe_img2img(
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
                image = pipe_img2img(
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
            "style": style,
            "character_name": features["name"],
            "ethnicity": features.get("ethnicity", "N/A"),
            "pose": pose_name,
            "quality": quality,
            "resolution": f"{final_w}x{final_h}",
            "generation_time": f"{gen_time:.2f}s"
        }
        
        # Supabase Storage
        if upload_cloud and SUPABASE_AVAILABLE:
            supabase_result = upload_to_supabase(image, features["name"], pose_name, seed, style)
            if supabase_result:
                result["image_url"] = supabase_result["url"]
                result["supabase_url"] = supabase_result["url"]
                result["supabase_path"] = supabase_result["path"]
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


runpod.serverless.start({"handler": handler})
