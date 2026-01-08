#!/usr/bin/env python3
"""
RunPod Serverless - NSFW Image Generator
Accepts Frontend payload format with character object
"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import runpod
import torch
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import time
import base64

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


# ============================================
# MODEL PATH
# ============================================
MODEL_PATH = "/app/models/civitai_new.safetensors"

# ============================================
# GLOBAL MODELS
# ============================================
pipe = None
pipe_img2img = None
compel = None
compel_img2img = None


# ============================================
# OCCUPATION SETTINGS
# ============================================
OCCUPATION_SETTINGS = {
    "Custom": {"background": "custom setting", "lighting": "natural lighting"},
    "None": {"background": "bedroom", "lighting": "soft lighting"},
    "Life Coach": {"background": "wellness office", "lighting": "warm light"},
    "Doctor": {"background": "medical office", "lighting": "bright lighting"},
    "Teacher": {"background": "classroom", "lighting": "bright lighting"},
    "Artist": {"background": "art studio", "lighting": "window light"},
    "Chef": {"background": "kitchen", "lighting": "bright lighting"},
    "Stripper": {"background": "club stage", "lighting": "neon lighting"},
    "Dominatrix": {"background": "dungeon", "lighting": "red lighting"},
    "Lawyer": {"background": "law office", "lighting": "office lighting"},
    "Engineer": {"background": "workshop", "lighting": "bright lighting"},
    "Professional Gamer": {"background": "gaming room with RGB lights", "lighting": "neon glow"},
    "Nurse": {"background": "hospital room", "lighting": "clinical lighting"},
    "Secretary": {"background": "office", "lighting": "office lighting"},
    "Maid": {"background": "luxury bedroom", "lighting": "soft lighting"},
    "Fitness Trainer": {"background": "gym", "lighting": "bright lighting"},
}


# ============================================
# QUALITY PRESETS
# ============================================
QUALITY_PRESETS = {
    "standard": {"base_width": 896, "base_height": 1152, "steps": 35, "cfg": 5.0, "highres_scale": 1.5, "highres_steps": 25, "highres_denoise": 0.4},
    "hd": {"base_width": 896, "base_height": 1152, "steps": 40, "cfg": 5.0, "highres_scale": 1.5, "highres_steps": 30, "highres_denoise": 0.45},
    "ultra_hd": {"base_width": 896, "base_height": 1152, "steps": 50, "cfg": 5.0, "highres_scale": 1.5, "highres_steps": 35, "highres_denoise": 0.45},
    "extreme": {"base_width": 896, "base_height": 1152, "steps": 60, "cfg": 5.0, "highres_scale": 1.5, "highres_steps": 40, "highres_denoise": 0.5},
}


# ============================================
# NEGATIVE PROMPTS
# ============================================
NEGATIVE_PROMPT = """score_6, score_5, score_4, (worst quality:1.4), (low quality:1.4),
bad anatomy, bad hands, extra fingers, missing fingers, ugly, deformed,
anime, cartoon, drawing, 3d render, cgi, plastic skin, blurry, watermark, text"""

SFW_NEGATIVE_PROMPT = NEGATIVE_PROMPT + """, nude, naked, NSFW, explicit, sexual, topless"""


# ============================================
# ALL PROMPTS - NSFW
# ============================================
PROMPTS = {
    # INTERCOURSE
    "doggy_style": """score_9, score_8_up, score_7_up, 1girl 1boy, doggystyle sex from behind, on all fours,
        {skin_description}, woman on hands and knees on bed, deeply arched back, ass raised high, head down moaning,
        {age} years old, {hair_description}, {eye_description}, gorgeous face showing pleasure,
        {breast_description} hanging swaying, athletic body, {butt_description} jiggling,
        pussy visible from behind stretched around cock, deep penetration, man kneeling behind gripping hips,
        thick penis penetrating deeply, intense thrusting, ass rippling from impact,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, masterpiece""",
    
    "missionary": """score_9, score_8_up, score_7_up, 1girl 1boy, missionary position sex, face to face,
        {skin_description}, woman lying on back on bed, legs spread wide, hair spread on pillow,
        {age} years old, {hair_description}, {eye_description}, face looking up with eye contact,
        {breast_description} visible, narrow waist, wide hips,
        pussy stretched around penetrating penis, deep penetration visible,
        athletic man on top between legs, arms supporting, thick penis penetrating,
        legs wrapped around waist, passionate intimate sex,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, masterpiece""",
    
    "cowgirl": """score_9, score_8_up, score_7_up, 1girl 1boy, cowgirl riding position, woman on top,
        {skin_description}, woman straddling riding, sitting on hips, knees on bed, thighs flexed,
        {age} years old, {hair_description} flowing with motion, {eye_description}, face tilted back moaning,
        {breast_description} bouncing wildly, athletic body, {butt_description},
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
    
    "pronebone": """score_9, score_8_up, score_7_up, 1girl 1boy, pronebone position, lying flat, pressed down,
        woman lying flat on stomach, face in pillow, ass slightly raised, body pinned,
        {age} years old, {hair_description} messy on pillow, {eye_description} glazed,
        {breast_description} squished flat, {skin_description} flushed sweating,
        pussy visible from behind, penetration from above, pinned position,
        man on top pressing down, deep grinding, body weight pressing, pinned helpless,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, intimate, masterpiece""",
    
    "against_wall": """score_9, score_8_up, score_7_up, 1girl 1boy, sex against wall, standing pressed, pinned,
        woman pressed against wall, legs wrapped around waist or standing, body pinned,
        {age} years old, {hair_description} messy, {eye_description} intense, gasping,
        {breast_description} pressed, {skin_description} flushed sweating,
        legs wrapped around waist, pussy penetrated, lifted off ground,
        strong man pressing into wall, hands under thighs, urgent hard thrusting,
        {setting_description}, photorealistic, 8K UHD, hyperrealistic, urgent, masterpiece""",

    # ORAL
    "blowjob": """score_9, score_8_up, score_7_up, 1girl 1boy, blowjob oral sex, cock in mouth, kneeling,
        {skin_description}, woman kneeling, face at crotch level, mouth wrapped around cock,
        {age} years old, {hair_description}, {eye_description} looking up eye contact,
        {breast_description} hanging, gorgeous face,
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
    
    "face_sitting": """score_9, score_8_up, score_7_up, 1girl 1boy, face sitting, sitting on face, dominant,
        woman sitting on man's face, straddling head, thighs around face, dominant,
        {age} years old, {hair_description}, {eye_description} showing dominance, moaning,
        {breast_description} nipples erect, {skin_description} glistening,
        pussy pressed on mouth, grinding on face, ass prominent,
        man lying flat, face covered, tongue working, grinding motion dominant pleasure,
        {setting_description}, photorealistic, 8K UHD, dominant, masterpiece""",
    
    "69_position": """score_9, score_8_up, score_7_up, 1girl 1boy, 69 position, mutual oral sex,
        two bodies in 69, woman on top inverted, simultaneous oral,
        {age} years old, {hair_description}, mutual pleasure expressions,
        {breast_description} hanging, {skin_description},
        mouth on penis sucking, pussy at his face being eaten, mutual stimulation,
        synchronized oral motion, both pleasuring each other simultaneously,
        {setting_description}, photorealistic, 8K UHD, mutual oral, masterpiece""",

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
    
    "vibrator": """score_9, score_8_up, score_7_up, 1girl solo, vibrator, using vibrator, magic wand,
        woman lying back, legs spread, using powerful vibrator,
        {age} years old, {hair_description} messy, {eye_description} rolled back, moaning,
        {breast_description}, {skin_description} sweating,
        legs spread, large vibrator pressed against pussy, vibrator on clit,
        body trembling, legs shaking, overwhelming pleasure from vibrator,
        {setting_description}, photorealistic, 8K UHD, vibrator visible, masterpiece""",

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
    
    "spread_ass": """score_9, score_8_up, score_7_up, 1girl solo, spread ass, hands spreading cheeks, explicit,
        woman bent over, hands reaching back spreading ass cheeks,
        {age} years old, {hair_description}, looking back submissive,
        athletic body from behind, {skin_description},
        hands gripping ass pulling apart, asshole fully exposed, anus prominent,
        pussy visible below, maximum spread, explicit display,
        {setting_description}, photorealistic, 8K UHD, explicit anal detail, masterpiece""",
    
    "spread_pussy": """score_9, score_8_up, score_7_up, 1girl solo, spreading pussy, spreading labia, explicit,
        woman lying back, legs spread wide, hands spreading labia,
        {age} years old, {hair_description}, {eye_description} seductive,
        {breast_description} nipples erect, {skin_description},
        legs maximally spread, fingers pulling labia apart, pussy opened fully,
        vaginal opening visible, pink inner walls, clit prominent, wetness,
        {setting_description}, photorealistic, 8K UHD, extreme genital detail, masterpiece""",

    # AFTERMATH
    "creampie": """score_9, score_8_up, score_7_up, 1girl, creampie cum dripping from pussy, aftermath,
        woman lying back, legs spread, relaxed post-sex, pussy exposed,
        {age} years old, {hair_description} messy, {eye_description} half-closed tired, satisfied,
        {breast_description}, {skin_description} glistening sweat,
        legs spread, pussy red used, thick white cum dripping out,
        semen flowing from vagina, creampie leaking, cum pooling, post-orgasm bliss,
        {setting_description}, photorealistic, 8K UHD, creampie detailed, masterpiece""",
    
    "cumshot": """score_9, score_8_up, score_7_up, 1girl 1boy, facial cumshot, cum on face,
        {skin_description}, woman kneeling, face tilted up, cum covering face,
        {age} years old, {hair_description} may have cum, {eye_description} looking up,
        {breast_description} may have cum, gorgeous face with white cum,
        face covered with cum, semen on cheeks nose forehead chin,
        thick ropes of cum, dripping, tongue out catching, mouth open,
        {setting_description}, photorealistic, 8K UHD, facial detailed, masterpiece""",
    
    "cumshot_face": """score_9, score_8_up, score_7_up, 1girl 1boy, facial cumshot, cum on face,
        {skin_description}, woman kneeling, face tilted up, cum covering face,
        {age} years old, {hair_description} may have cum, {eye_description} looking up,
        {breast_description} may have cum, gorgeous face with white cum,
        face covered with cum, semen on cheeks nose forehead chin,
        thick ropes of cum, dripping, tongue out catching, mouth open,
        {setting_description}, photorealistic, 8K UHD, facial detailed, masterpiece""",
    
    "cum_on_body": """score_9, score_8_up, score_7_up, 1girl 1boy, cum on body, semen on skin,
        woman lying exposed, post-sex, cum visible on body,
        {age} years old, {hair_description} messy, {eye_description} satisfied,
        {breast_description} with cum, {skin_description} with white cum,
        cum on breasts belly thighs, semen pooled on stomach, ropes across chest,
        man nearby penis dripping, body cumshot aftermath,
        {setting_description}, photorealistic, 8K UHD, cum visible clearly, masterpiece""",
    
    "bukkake": """score_9, score_8_up, score_7_up, 1girl multiple boys, bukkake, multiple cumshots,
        woman kneeling surrounded by men, face and body exposed,
        {age} years old, {hair_description} soaked with cum, eyes barely visible,
        {breast_description} covered in cum, {skin_description} covered in semen,
        face completely covered in cum, multiple loads, hair soaked, body drenched,
        multiple men around, some still cumming, overwhelmed by cum,
        group setting, photorealistic, 8K UHD, bukkake detailed, masterpiece""",

    # BDSM
    "handcuffs": """score_9, score_8_up, score_7_up, 1girl, handcuffs wrists bound, bondage light,
        woman standing or kneeling, hands cuffed behind back, vulnerable,
        {age} years old, {hair_description}, {eye_description} submissive,
        {breast_description} thrust forward, {skin_description},
        hands cuffed behind back, metal handcuffs visible, unable to cover self,
        restrained helpless, submissive bondage,
        {setting_description}, photorealistic, 8K UHD, handcuffs detailed, masterpiece""",
    
    "collar_leash": """score_9, score_8_up, score_7_up, 1girl 1boy, collar and leash, pet play BDSM,
        woman on all fours or kneeling, collar around neck, leash held by man,
        {age} years old, {hair_description}, {eye_description} looking up obedient,
        {breast_description} hanging, {skin_description},
        leather collar tight around neck, leash attached, being led,
        dominant man holding leash, submissive pet, collared and owned,
        {setting_description}, photorealistic, 8K UHD, collar detailed, masterpiece""",
    
    "blindfolded": """score_9, score_8_up, score_7_up, 1girl, blindfolded eyes covered, sensory play,
        woman lying or standing, eyes covered by blindfold, body exposed,
        {age} years old, {hair_description} visible around blindfold, mouth slightly open,
        {breast_description} nipples erect, {skin_description},
        black silk blindfold over eyes, cannot see, vulnerable, heightened senses,
        anticipating touch, sensory deprivation,
        {setting_description}, photorealistic, 8K UHD, blindfold detailed, masterpiece""",

    # GROUP
    "gangbang": """score_9, score_8_up, score_7_up, 1girl multiple boys, gangbang group sex,
        woman surrounded by multiple men, multiple cocks, being fucked,
        {age} years old, {hair_description} messy wild, {eye_description} glazed overwhelmed,
        {breast_description} being groped, {skin_description} sweating,
        multiple penetrations, cock in pussy mouth hands, surrounded,
        multiple men visible, overwhelmed by multiple, group fucking,
        {setting_description}, photorealistic, 8K UHD, group scene, masterpiece""",
    
    "double_penetration": """score_9, score_8_up, score_7_up, 1girl 2boys, double penetration DP,
        woman sandwiched between two men, both holes filled,
        {age} years old, {hair_description} wild, {eye_description} overwhelmed, mouth gaped,
        {breast_description} pressed, {skin_description} sweating,
        cock in pussy, cock in ass, both holes penetrated simultaneously,
        two men, one behind one in front, extreme fullness, DP intensity,
        {setting_description}, photorealistic, 8K UHD, DP detailed, masterpiece""",
    
    "threesome": """score_9, score_8_up, score_7_up, 1girl 2boys, MFM threesome,
        woman with two men, cock in mouth, cock in pussy, spit-roasted,
        {age} years old, {hair_description}, {eye_description} focused,
        {breast_description} hanging bouncing, {skin_description},
        mouth on penis sucking, pussy penetrated, two cocks serviced,
        two men visible, pleasuring both simultaneously, MFM threesome,
        {setting_description}, photorealistic, 8K UHD, threesome detailed, masterpiece""",
    
    "lesbian": """score_9, score_8_up, score_7_up, 2girls, lesbian sex, girl on girl,
        two women together, bodies intertwined, intimate lesbian position,
        both {age} years old, {hair_description} on both, passionate expressions,
        {breast_description} pressed together, {skin_description} touching,
        breasts against breasts, pussies touching or tribbing, mutual pleasure,
        hands exploring, lesbian intimacy,
        {setting_description}, photorealistic, 8K UHD, lesbian intimate, masterpiece""",

    # MISC
    "standing": """score_9, score_8_up, score_7_up, 1girl solo, standing nude, confident pose,
        {skin_description}, woman standing upright, nude, legs slightly apart, confident,
        {age} years old, {hair_description}, {eye_description} looking at camera,
        {breast_description} natural position, {butt_description},
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
    
    "showering": """score_9, score_8_up, score_7_up, 1girl solo, showering wet, water running,
        woman standing in shower, water cascading over body, completely wet,
        {age} years old, {hair_description} completely wet slicked, {eye_description},
        {breast_description} wet water streaming, {skin_description} glistening wet,
        standing in shower, water running over body, steam surrounding,
        arms raised washing hair, water droplets everywhere,
        shower setting, photorealistic, 8K UHD, wet skin detailed, masterpiece""",
    
    "bath": """score_9, score_8_up, score_7_up, 1girl solo, bathing in tub, bubbles, relaxed,
        woman relaxing in bathtub, body submerged, bubbles covering,
        {age} years old, {hair_description} up in bun, {eye_description} peaceful,
        {breast_description} partially visible above water, {skin_description} wet,
        sitting in warm bath, bubbles on surface, comfortable relaxed,
        arms on tub edge, peaceful bath moment,
        bathroom, candles, warm lighting, photorealistic, 8K UHD, cozy, masterpiece""",
    
    "yoga_pose": """score_9, score_8_up, score_7_up, 1girl solo, yoga pose downward dog, flexible nude,
        woman in downward dog yoga position, inverted V, hips raised,
        {age} years old, {hair_description} in bun, {eye_description} focused,
        {breast_description} hanging, {skin_description} light sweat,
        hands and feet on floor, hips raised high, ass prominent,
        pussy visible from behind, yoga nude flexibility,
        yoga studio, mat, natural lighting, photorealistic, 8K UHD, athletic, masterpiece""",
    
    "jack_o_pose": """score_9, score_8_up, score_7_up, 1girl solo, jack-o pose, extreme arch, flexibility,
        woman in extreme jack-o pose, face down, ass raised maximally high,
        {age} years old, {hair_description}, athletic flexible body,
        {breast_description} pressed on floor, {skin_description},
        chest flat on floor, back arched extreme, ass raised high,
        pussy visible from behind, extreme flexibility display,
        {setting_description}, photorealistic, 8K UHD, extreme flexibility, masterpiece""",
}


# ============================================
# SFW PROMPTS
# ============================================
SFW_PROMPTS = {
    "standing": """score_9, score_8_up, score_7_up, 1girl solo, standing confident, full body,
        woman standing upright, natural pose, relaxed stance,
        {age} years old, {hair_description}, {eye_description}, natural smile,
        {breast_description}, {skin_description}, casual clothes,
        {setting_description}, photorealistic, 8K UHD, professional, masterpiece""",
    
    "sitting": """score_9, score_8_up, score_7_up, 1girl solo, sitting relaxed, full body,
        woman sitting comfortably, natural seated position,
        {age} years old, {hair_description}, {eye_description}, gentle smile,
        {breast_description}, {skin_description}, casual attire,
        {setting_description}, photorealistic, 8K UHD, professional, masterpiece""",
    
    "portrait": """score_9, score_8_up, score_7_up, 1girl solo, portrait headshot, close-up,
        woman facing camera, professional portrait,
        {age} years old, {hair_description}, {eye_description}, natural smile,
        {skin_description}, professional headshot, confident,
        studio portrait, professional lighting,
        photorealistic, 8K UHD, professional photography, masterpiece""",
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
    
    print("✅ Model loaded!")
    return pipe, pipe_img2img


# ============================================
# HELPER FUNCTIONS
# ============================================
def safe_format(template: str, **kwargs) -> str:
    """Format string with fallback for missing keys"""
    import re
    # Find all {key} placeholders
    placeholders = re.findall(r'\{(\w+)\}', template)
    # Add default empty string for missing keys
    for key in placeholders:
        if key not in kwargs:
            kwargs[key] = ""
    return template.format(**kwargs)


def split_prompt_for_sdxl(prompt: str) -> tuple:
    parts = [p.strip() for p in prompt.split(',') if p.strip()]
    if len(parts) <= 30:
        return (prompt, prompt)
    
    style_keywords = ['photorealistic', 'realistic', '8k', 'uhd', 'masterpiece', 'detailed', 'lighting']
    prompt_1_parts, prompt_2_parts = [], []
    
    for part in parts:
        if any(kw in part.lower() for kw in style_keywords):
            prompt_2_parts.append(part)
        else:
            prompt_1_parts.append(part)
    
    return (', '.join(prompt_1_parts[:35]), ', '.join(prompt_2_parts[:35]))


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
    # Extract hair info
    hair_style = character.get("hairStyle", "long").lower()
    hair_colors = ['blonde', 'brunette', 'red', 'black', 'brown', 'white', 'pink', 'blue', 'auburn', 'ginger', 'silver', 'purple']
    hair_color = None
    description = character.get("description", "").lower()
    for color in hair_colors:
        if color in description:
            hair_color = color
            break
    hair_description = f"{hair_color or ''} {hair_style} hair".strip()
    
    # Eye color
    eye_color = character.get("eyeColor", "blue").lower()
    eye_description = f"{eye_color} eyes"
    
    # Breast size
    breast_size = character.get("breastSize", "Large").lower()
    breast_map = {
        "small": "small breasts, petite chest",
        "medium": "medium breasts, average chest",
        "large": "large breasts, big chest",
        "extra-large": "huge breasts, massive chest"
    }
    breast_description = breast_map.get(breast_size, "large breasts")
    
    # Butt size
    butt_size = character.get("buttSize", "Medium").lower()
    butt_map = {
        "small": "small butt, petite ass",
        "medium": "round butt, medium ass",
        "large": "big butt, thick ass, bubble butt",
        "extra-large": "huge butt, massive ass, thicc"
    }
    butt_description = butt_map.get(butt_size, "round butt")
    
    # Ethnicity - DETAILED mapping for accurate representation
    ethnicity = character.get("ethnicity", "white").lower()
    
    # Detailed ethnicity descriptions for SDXL - Matching App Options
    ethnicity_map = {
        # === REAL ETHNICITIES ===
        "asian": {
            "skin": "fair porcelain skin",
            "features": "east asian woman, asian face, monolid eyes, delicate asian features, beautiful asian girl",
            "body": "petite asian body, slim figure"
        },
        "black": {
            "skin": "dark brown skin, ebony skin, rich dark complexion",
            "features": "black woman, african features, beautiful black girl, full lips, african american",
            "body": "curvy body, thick thighs"
        },
        "white": {
            "skin": "fair skin, pale porcelain skin",
            "features": "caucasian woman, european features, beautiful white girl, western face",
            "body": "athletic body"
        },
        "latina": {
            "skin": "tan skin, olive skin, caramel complexion, sun-kissed skin",
            "features": "latina woman, latin features, beautiful latina girl, hispanic, latin american",
            "body": "curvy latina body, wide hips, hourglass figure"
        },
        "arab": {
            "skin": "olive skin, tan skin, middle eastern complexion",
            "features": "arab woman, middle eastern features, beautiful arab girl, exotic middle eastern beauty, persian features",
            "body": "curvy body"
        },
        "indian": {
            "skin": "brown skin, caramel skin, warm brown complexion",
            "features": "indian woman, south asian features, beautiful indian girl, desi beauty, bollywood beauty",
            "body": "curvy body, wide hips"
        },
        
        # === FANTASY ETHNICITIES (PRO) ===
        "elf": {
            "skin": "pale ethereal skin, flawless porcelain skin, luminous skin",
            "features": "elf woman, elven features, pointed elf ears, ethereal beauty, fantasy elf, high elf, elegant elven face, otherworldly beauty",
            "body": "slender elegant body, graceful elven figure"
        },
        "alien": {
            "skin": "pale pink skin, unusual skin tone, otherworldly complexion",
            "features": "alien woman, alien humanoid, pointed ears, exotic alien features, extraterrestrial beauty, sci-fi alien girl, unusual eye color",
            "body": "slim otherworldly body"
        },
        "demon": {
            "skin": "pale skin with dark undertones, supernatural complexion",
            "features": "demon woman, demon girl, succubus, small horns on head, demonic beauty, supernatural features, seductive demon, dark fantasy",
            "body": "curvy seductive body, supernatural figure"
        },
        
        # === FALLBACK ALIASES ===
        "european": {
            "skin": "fair skin",
            "features": "european woman, western features",
            "body": "athletic body"
        },
        "african": {
            "skin": "dark ebony skin",
            "features": "african woman, african features",
            "body": "curvy body"
        },
        "hispanic": {
            "skin": "tan skin, olive skin",
            "features": "hispanic woman, latin features",
            "body": "curvy body"
        },
        "middle_eastern": {
            "skin": "olive skin",
            "features": "middle eastern woman, persian features",
            "body": "curvy body"
        },
        "mixed": {
            "skin": "tan skin",
            "features": "mixed race woman, exotic beauty, unique features",
            "body": "athletic body"
        }
    }
    
    # Get ethnicity details or default to white
    eth_details = ethnicity_map.get(ethnicity, ethnicity_map["white"])
    
    # Combine skin and features for skin_description
    skin_description = f"{eth_details['skin']}, {eth_details['features']}"
    
    # Add body type from ethnicity if not conflicting
    ethnicity_body = eth_details.get('body', '')
    
    # Occupation/setting
    personality = character.get("personalityId", {}) or {}
    occupation = personality.get("occupationId", "None")
    occ_setting = OCCUPATION_SETTINGS.get(occupation, OCCUPATION_SETTINGS["None"])
    setting_description = f"{occ_setting['background']}, {occ_setting['lighting']}"
    
    return {
        "age": character.get("age", 25),
        "name": character.get("name", "unknown"),
        "gender": character.get("gender", "Female").lower(),
        "ethnicity": ethnicity,
        "hair_description": hair_description,
        "eye_description": eye_description,
        "breast_description": breast_description,
        "butt_description": butt_description,
        "skin_description": skin_description,
        "ethnicity_body": ethnicity_body,
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
        
        # Parse character data
        character = input_data.get("character", {})
        if character:
            features = parse_character(character)
        else:
            # Fallback to flat params
            features = {
                "age": input_data.get("age", 25),
                "name": input_data.get("character_name", "unknown"),
                "ethnicity": input_data.get("ethnicity", "white"),
                "hair_description": input_data.get("hair_description", "long blonde hair"),
                "eye_description": input_data.get("eye_description", "blue eyes"),
                "breast_description": input_data.get("breast_description", "large breasts"),
                "butt_description": input_data.get("butt_description", "round butt"),
                "skin_description": input_data.get("skin_description", "fair skin, caucasian features"),
                "setting_description": input_data.get("setting_description", "bedroom, soft lighting"),
            }
        
        # Get pose - check multiple sources
        pose_name = input_data.get("pose_name", "")
        if not pose_name and character:
            personality = character.get("personalityId", {}) or {}
            pose_name = personality.get("poseId", "standing")
        if not pose_name:
            pose_name = "standing"
        
        # Normalize pose name
        pose_name = pose_name.lower().replace(' ', '_').replace('-', '_')
        
        # Other params
        quality = input_data.get("quality", "ultra_hd")
        seed = input_data.get("seed") or -1
        use_highres = input_data.get("use_highres", True)
        enhance_img = input_data.get("enhance", True)
        nsfw = input_data.get("nsfw", True)
        upload_cloud = input_data.get("upload_cloudinary", True)
        
        # Select prompt dictionary
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
        
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["ultra_hd"])
        
        if seed == -1 or seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"\n🎨 Image: {features['name']} | Pose: {pose_name} | Quality: {quality}")
        start_time = time.time()
        
        # Load model
        model, model_img2img = load_models()
        
        # Generate
        prompt_1, prompt_2 = split_prompt_for_sdxl(prompt)
        neg_1, neg_2 = split_prompt_for_sdxl(negative_prompt)
        
        image = model(
            prompt=prompt_1, prompt_2=prompt_2,
            negative_prompt=neg_1, negative_prompt_2=neg_2,
            width=preset['base_width'], height=preset['base_height'],
            num_inference_steps=preset['steps'], guidance_scale=preset['cfg'],
            generator=generator, clip_skip=2
        ).images[0]
        
        # Highres
        if use_highres:
            final_w = int(preset['base_width'] * preset['highres_scale'])
            final_h = int(preset['base_height'] * preset['highres_scale'])
            upscaled = image.resize((final_w, final_h), Image.LANCZOS)
            generator = torch.Generator(device="cuda").manual_seed(seed + 1)
            
            image = model_img2img(
                prompt=prompt_1, prompt_2=prompt_2,
                negative_prompt=neg_1, negative_prompt_2=neg_2,
                image=upscaled, strength=preset['highres_denoise'],
                num_inference_steps=preset['highres_steps'], guidance_scale=preset['cfg'],
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
