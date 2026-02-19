import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

def generate_ocr_sample(text, font_path, font_size=40, img_size=(300, 80)):
    # 1. Create a base white image
    img = Image.new('RGB', img_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 2. Load font and render text
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print("Font file not found. Using default.")
        font = ImageFont.load_default()

    # Center the text
    w, h = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (100, 40) # Fallback for newer PIL
    draw.text(((img_size[0]-w)/2, (img_size[1]-h)/2), text, fill=(0, 0, 0), font=font)

    # 3. Generate Colored Gaussian Noise
    # Convert image to numpy array
    data = np.array(img).astype(np.float32)
    
    # Create noise: mean=0, sigma=25 (adjust sigma for more/less noise)
    # Shape is (height, width, 3) for RGB
    noise = np.random.normal(0, 25, data.shape)
    
    # 4. Add noise to the image
    noisy_data = data + noise
    
    # Clip values to stay in [0, 255] range and convert back to uint8
    noisy_data = np.clip(noisy_data, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_data)

# --- Usage Example ---
my_text = "SYNTH-OCR-2026"
# Replace with a path to a font on your system (e.g., Arial.ttf)
sample_img = generate_ocr_sample(my_text, "arial.ttf")
sample_img.save("synthetic_data.png")
sample_img.show()