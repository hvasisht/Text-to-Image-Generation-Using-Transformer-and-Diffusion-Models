import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime
from textblob import TextBlob
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Page config
st.set_page_config(
    page_title="Text-to-Image Generator",
    page_icon="🎨",
    layout="centered"
)

# ============================================
# HELPER FUNCTIONS
# ============================================

def fix_spelling(text):
    """Fix spelling mistakes in prompt"""
    blob = TextBlob(text)
    corrected = str(blob.correct())
    return corrected

@st.cache_resource
def load_diffusion_model():
    """Load Stable Diffusion model (cached)"""
    with st.spinner("Loading Stable Diffusion model... (first time only)"):
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = pipe.to(device)
    return pipe

@st.cache_resource
def load_clip_model():
    """Load CLIP model for similarity scoring (cached)"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device

def calculate_clip_score(image, prompt):
    """Calculate CLIP similarity between image and prompt"""
    model, processor, device = load_clip_model()
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    similarity = outputs.logits_per_image[0][0].item()
    return similarity

# ============================================
# STYLE PRESETS
# ============================================

STYLE_PRESETS = {
    "None": "",
    "Photorealistic": ", highly detailed, 8K resolution, photorealistic, professional photography",
    "Oil Painting": ", oil painting style, artistic, textured brushstrokes, classical art",
    "Watercolor": ", watercolor painting, soft colors, artistic, delicate",
    "Anime": ", anime style, vibrant colors, manga art, Japanese animation",
    "Sketch": ", pencil sketch, black and white drawing, artistic linework",
    "Cyberpunk": ", cyberpunk style, neon colors, futuristic, dystopian",
    "Fantasy Art": ", fantasy art, magical, ethereal, dreamlike, mystical"
}

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

# ============================================
# MAIN UI
# ============================================

# Title and description
st.title("🎨 Advanced Text-to-Image Generator")
st.markdown("**IE 7615 Deep Learning for AI - Generative Project**")
st.markdown("*Enhanced with Spell Correction, Style Presets, Negative Prompts & Quality Metrics*")
st.divider()

# ============================================
# SIDEBAR - SETTINGS
# ============================================

st.sidebar.header("⚙️ Generation Settings")

# Basic settings
cfg_scale = st.sidebar.slider(
    "CFG Scale",
    min_value=1.0,
    max_value=15.0,
    value=7.5,
    step=0.5,
    help="How closely to follow the prompt (7.5 is optimal)"
)

num_steps = st.sidebar.slider(
    "Inference Steps",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
    help="More steps = better quality but slower"
)

# Style presets
st.sidebar.subheader("🎨 Style Presets")
selected_style = st.sidebar.selectbox(
    "Apply artistic style:",
    list(STYLE_PRESETS.keys()),
    help="Add stylistic modifiers to your prompt"
)

# Batch generation
st.sidebar.subheader("🔢 Batch Generation")
num_variations = st.sidebar.slider(
    "Number of variations",
    min_value=1,
    max_value=4,
    value=1,
    help="Generate multiple variations of the same prompt"
)

# Features toggles
st.sidebar.subheader("✨ Smart Features")
enable_spell_check = st.sidebar.checkbox("Auto-correct spelling", value=True)
show_clip_score = st.sidebar.checkbox("Show quality score", value=True)

st.sidebar.divider()

# Project info
st.sidebar.markdown("**Group 9:**")
st.sidebar.info(
    """
    - Harini Vasisht
    - Samruddhi Bansod
    - Pranav Rangbulla
    - Dhanush Manohoran
    
    **Metrics Achieved:**
    - FID: 374.47
    - IS: 5.08
    - CLIP: 31.85
    """
)

# ============================================
# MAIN INTERFACE
# ============================================

st.subheader("📝 Enter Your Prompt")

# Main prompt input
prompt = st.text_area(
    "Describe the image you want to generate:",
    placeholder="Example: A serene Japanese garden with a red bridge over a koi pond",
    height=100,
    key="main_prompt"
)

# Negative prompt
with st.expander("🚫 Advanced: Negative Prompt (Optional)", expanded=False):
    st.markdown("Specify what you **don't** want in the image:")
    negative_prompt = st.text_input(
        "Negative prompt:",
        placeholder="blurry, cartoon, low quality, watermark, text",
        help="Elements to avoid in generation"
    )

# Example prompts
st.markdown("**✨ Try these examples:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🦁 Lion", use_container_width=True):
        st.session_state.main_prompt = "A majestic lion in the African savanna"
        st.rerun()
with col2:
    if st.button("🌅 Sunset", use_container_width=True):
        st.session_state.main_prompt = "A beautiful sunset over the ocean"
        st.rerun()
with col3:
    if st.button("🏰 Castle", use_container_width=True):
        st.session_state.main_prompt = "A medieval castle on a cliff"
        st.rerun()
with col4:
    if st.button("🌸 Garden", use_container_width=True):
        st.session_state.main_prompt = "A peaceful zen garden with cherry blossoms"
        st.rerun()

# Recent prompts
if st.session_state.prompt_history:
    with st.expander("📜 Recent Prompts", expanded=False):
        for i, old_prompt in enumerate(reversed(st.session_state.prompt_history[-5:])):
            if st.button(f"↻ {old_prompt[:50]}...", key=f"history_{i}"):
                st.session_state.main_prompt = old_prompt
                st.rerun()

st.divider()

# ============================================
# GENERATE BUTTON
# ============================================

if st.button("🎨 Generate Image", type="primary", use_container_width=True):
    if not prompt:
        st.warning("⚠️ Please enter a prompt first!")
    else:
        # Save to history
        if prompt not in st.session_state.prompt_history:
            st.session_state.prompt_history.append(prompt)
        
        # Spell correction
        original_prompt = prompt
        if enable_spell_check:
            corrected_prompt = fix_spelling(prompt)
            if corrected_prompt != prompt:
                st.info(f"✨ **Auto-corrected:** '{prompt}' → '{corrected_prompt}'")
                prompt = corrected_prompt
        
        # Apply style preset
        if selected_style != "None":
            prompt = prompt + STYLE_PRESETS[selected_style]
            st.info(f"🎨 **Applied style:** {selected_style}")
        
        # Load model
        pipe = load_diffusion_model()
        
        # Generate images
        if num_variations == 1:
            # Single image generation
            with st.spinner(f"🎨 Generating image... (takes ~30-40 seconds)"):
                start_time = datetime.now()
                
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=num_steps,
                    guidance_scale=cfg_scale
                ).images[0]
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
            
            # Display results
            st.success(f"✅ Image generated in {duration:.1f} seconds!")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(image, caption=f"Prompt: {original_prompt}", use_container_width=True)
            
            with col2:
                st.metric("⏱️ Time", f"{duration:.1f}s")
                st.metric("⚙️ CFG", cfg_scale)
                st.metric("🔢 Steps", num_steps)
                
                # CLIP Score
                if show_clip_score:
                    with st.spinner("Calculating quality..."):
                        clip_score = calculate_clip_score(image, original_prompt)
                    st.metric("🎯 Match Score", f"{clip_score:.2f}")
                    
                    if clip_score > 30:
                        st.success("Excellent!")
                    elif clip_score > 25:
                        st.info("Good match")
                    else:
                        st.warning("Consider refining")
                
                # Download button
                from io import BytesIO
                buf = BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download",
                    data=buf.getvalue(),
                    file_name=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        else:
            # Batch generation
            st.info(f"🔢 Generating {num_variations} variations...")
            
            cols = st.columns(min(num_variations, 2))
            
            for i in range(num_variations):
                with cols[i % 2]:
                    with st.spinner(f"Generating #{i+1}..."):
                        start_time = datetime.now()
                        
                        image = pipe(
                            prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            num_inference_steps=num_steps,
                            guidance_scale=cfg_scale
                        ).images[0]
                        
                        duration = (end_time - start_time).total_seconds()
                    
                    st.image(image, caption=f"Variation {i+1} ({duration:.1f}s)")
                    
                    # Mini download button
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        label=f"⬇️ Download #{i+1}",
                        data=buf.getvalue(),
                        file_name=f"var_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        key=f"download_{i}",
                        use_container_width=True
                    )
            
            st.success(f"✅ All {num_variations} variations generated!")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <b>Enhanced Features:</b> Spell Correction • Style Presets • Negative Prompts • Quality Metrics • Batch Generation<br>
    Northeastern University | IE 7615 Deep Learning for AI
    </div>
    """,
    unsafe_allow_html=True
)