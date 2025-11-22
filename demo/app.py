import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Text-to-Image Generator",
    page_icon="🎨",
    layout="centered"
)

# Title and description
st.title("🎨 Text-to-Image Generator")
st.markdown("**IE 7615 Deep Learning for AI - Generative Project**")
st.markdown("*Using CLIP + Stable Diffusion v1.5*")
st.divider()

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    with st.spinner("Loading Stable Diffusion model... (first time only)"):
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = pipe.to(device)
    return pipe

# Sidebar for settings
st.sidebar.header("⚙️ Generation Settings")
st.sidebar.markdown("**Optimal settings from Milestone 2:**")

cfg_scale = st.sidebar.slider(
    "CFG Scale",
    min_value=1.0,
    max_value=15.0,
    value=7.5,
    step=0.5,
    help="Guidance scale - how closely to follow the prompt"
)

num_steps = st.sidebar.slider(
    "Inference Steps",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
    help="Number of denoising steps"
)

st.sidebar.divider()
st.sidebar.markdown("**Project Info:**")
st.sidebar.info(
    """
    **Group 9:**
    - Harini Vasisht
    - Samruddhi Bansod
    - Pranav Rangbulla
    - Dhanush Manohoran
    
    **Metrics Achieved:**
    - FID: 374.47
    - IS: 5.08
    - CLIP: 31.85
    
    **Optimal Settings:**
    - CFG: 7.5
    - Steps: 20
    """
)

# Main interface
st.subheader("Enter Your Prompt")
prompt = st.text_area(
    "Describe the image you want to generate:",
    placeholder="Example: A serene Japanese garden with a red bridge over a koi pond",
    height=100
)

# Example prompts
st.markdown("**Try these examples:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🦁 Lion"):
        prompt = "A majestic lion in the African savanna"
with col2:
    if st.button("🌅 Sunset"):
        prompt = "A beautiful sunset over the ocean"
with col3:
    if st.button("🏰 Castle"):
        prompt = "A medieval castle on a cliff"

st.divider()

# Generate button
if st.button("🎨 Generate Image", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a prompt first!")
    else:
        # Load model
        pipe = load_model()
        
        # Generate
        with st.spinner(f"Generating image... (this takes ~30-40 seconds)"):
            start_time = datetime.now()
            
            image = pipe(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=cfg_scale
            ).images[0]
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
        
        # Display results
        st.success(f"✅ Image generated in {duration:.1f} seconds!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption=f"Prompt: {prompt}", use_container_width=True)
        
        with col2:
            st.metric("Generation Time", f"{duration:.1f}s")
            st.metric("CFG Scale", cfg_scale)
            st.metric("Steps", num_steps)
            
            # Download button
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Image",
                data=buf.getvalue(),
                file_name=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Northeastern University | IE 7615 Deep Learning for AI<br>
    Text-to-Image Generation Using Transformer and Diffusion Models
    </div>
    """,
    unsafe_allow_html=True
)