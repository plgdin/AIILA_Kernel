import cv2
import numpy as np

class UIHub:
    def __init__(self, width=1000, height=700):
        self.w, self.h = width, height
        self.accent_color = (255, 191, 0) # Cyber Blue

    def create_base_frame(self):
        # Create dark semi-transparent background
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15) # Deep matte gray
        
        # 1. Top Header Bar
        cv2.rectangle(canvas, (0, 0), (self.w, 50), (30, 30, 30), -1)
        cv2.putText(canvas, "AIILA | HOLOMAT CORE v2.0", (20, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.accent_color, 2)

        # 2. Left Toolbox Rail
        cv2.rectangle(canvas, (0, 50), (180, self.h), (25, 25, 25), -1)
        cv2.line(canvas, (180, 50), (180, self.h), (50, 50, 50), 1)

        # 3. Bottom Terminal/Status Bar
        cv2.rectangle(canvas, (180, self.h-100), (self.w, self.h), (20, 20, 20), -1)
        cv2.line(canvas, (180, self.h-100), (self.w, self.h-100), (50, 50, 50), 1)
        
        return canvas

    def update_status(self, canvas, message):
        cv2.putText(canvas, f"> SYSTEM_LOG: {message}", (200, self.h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)