import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

class UIHub:
    def __init__(self, width=1000, height=700):
        self.w, self.h = width, height
        self.accent_color = (255, 191, 0) 
        # Path to your fonts folder
        self.font_bold = "fonts/arialbd.ttf"
        self.font_regular = "fonts/consola.ttf"

    def draw_modern_text(self, img, text, position, font_path, font_size, color):
        """Helper to render high-quality text on the canvas."""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        # PIL uses RGB (color[2], color[1], color[0] converts BGR to RGB)
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def create_base_frame(self):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15) 
        
        # 1. Top Header Bar
        cv2.rectangle(canvas, (0, 0), (self.w, 50), (30, 30, 30), -1)
        
        # DRAW MODERN TITLE
        canvas = self.draw_modern_text(
            canvas, 
            "AIILA | HOLOMAT CORE v2.0", 
            (180, 12), 
            self.font_bold, 22, self.accent_color
        )

        # 2. Left Toolbox Rail
        cv2.rectangle(canvas, (0, 50), (180, self.h), (25, 25, 25), -1)
        cv2.line(canvas, (180, 50), (180, self.h), (50, 50, 50), 1)

        # 3. Bottom Terminal
        cv2.rectangle(canvas, (180, self.h-100), (self.w, self.h), (20, 20, 20), -1)
        cv2.line(canvas, (180, self.h-100), (self.w, self.h-100), (50, 50, 50), 1)
        
        return canvas

    def update_status(self, canvas, message):
        # DRAW MODERN SYSTEM LOG
        return self.draw_modern_text(
            canvas, 
            f"> SYSTEM_LOG: {message}", 
            (200, self.h - 60), 
            self.font_regular, 16, (0, 255, 0)
        )