import cv2
import numpy as np

def assemble_final_os_view(projector_canvas, live_camera_frame):
    """
    Creates the 'Big Screen + Small Feed' layout.
    - Big Rectangle (Left): The AR Projector (1000x700)
    - Small Rectangle (Right): The Live Camera Feed (320x240)
    """
    # Create a master slate (1400x700)
    # 1400 width gives enough room for 1000px (projector) + 400px (sidebar area)
    master_frame = np.zeros((700, 1400, 3), dtype=np.uint8)
    master_frame[:] = (18, 18, 18) # Dark aesthetic background

    # 1. Place the Projector Canvas (The Big Screen on the Left)
    # We maintain 1000x700 as the primary workspace
    proj_h, proj_w = projector_canvas.shape[:2]
    master_frame[0:proj_h, 0:proj_w] = projector_canvas

    # 2. Place the Live Camera Feed (Small Rectangle on the Right)
    # We resize the raw camera frame to a small 'PiP' window
    cam_small = cv2.resize(live_camera_frame, (320, 240))
    
    # Position it in the middle of the remaining right side
    # X start = 1050 (Leaving a 50px gap from the projector)
    # Y start = 230 (Centered vertically)
    x_pos, y_pos = 1050, 230
    
    # Draw a tech-border around the camera feed
    cv2.rectangle(master_frame, (x_pos-2, y_pos-2), (x_pos+322, y_pos+242), (0, 255, 0), 1)
    master_frame[y_pos:y_pos+240, x_pos:x_pos+320] = cam_small

    return master_frame