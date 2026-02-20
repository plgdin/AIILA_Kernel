import cv2
from PIL import Image
from google import genai

# Setup Gemini API
client = genai.Client(api_key="AIzaSyB1f6Pe348ThuMGf4fOccBOUMdHPD4yQ1Y")

def scan_object(frame):
    print("Scanning with Gemini Vision Core...")
    # Convert OpenCV BGR frame to RGB for Gemini
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    try:
        # The "Pipe Split" Prompt
        prompt = """
        Identify the object in the image. I need the broad category AND the exact specific model name.
        Reply strictly in this format: category | exact model name
        Valid categories are: 'smartphone', 'calculator', or 'unknown'. 
        Example 1: smartphone | Nothing Phone 3a Pro
        Example 2: calculator | Casio fx-991EX
        Example 3: unknown | Sony WH-1000XM4 Headphones
        Do not add any other text, markdown, or formatting.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_img]
        )
        ans = response.text.strip().replace("```", "").replace("\n", "")
        
        # Split the AI's answer into two variables
        if "|" in ans:
            category, model = ans.split("|", 1)
            print(f"AI Identified Category: {category.strip()}")
            print(f"AI Identified Exact Model: {model.strip()}")
            return category.strip().lower(), model.strip().title()
        else:
            print(f"AI Raw Answer: {ans}")
            return "unknown", ans.strip()
            
    except Exception as e:
        print(f"API Error: {e}")
        return "error", "error"