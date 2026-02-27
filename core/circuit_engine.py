import cv2
import numpy as np

class CircuitEngine:
    def __init__(self):
        self.placed_components = []
        self.catalog = {
            "resistor": {"color": (0, 255, 0), "label": "RES 10k"},
            "led": {"color": (0, 0, 255), "label": "LED RED"},
            "mcu": {"color": (255, 100, 0), "label": "ESP32"}
        }

    def add_component(self, type_name, x, y):
        if type_name in self.catalog:
            comp = {
                "type": type_name,
                "pos": (x, y),
                "data": self.catalog[type_name]
            }
            self.placed_components.append(comp)

    def draw_components(self, canvas):
        for comp in self.placed_components:
            x, y = comp['pos']
            color = comp['data']['color']
            
            # Draw Holographic "Glow" Symbol
            cv2.circle(canvas, (x, y), 10, color, -1)
            cv2.circle(canvas, (x, y), 15, color, 2) # Outer glow ring
            cv2.putText(canvas, comp['data']['label'], (x + 20, y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- NEW METHOD TO CLEAR MARKED COMPONENTS ---
    def clear_components(self):
        """Wipes all placed components from the storage list."""
        self.placed_components = []