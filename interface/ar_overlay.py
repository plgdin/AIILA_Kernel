import cv2
import numpy as np

def assemble_final_os_view(projector_canvas, live_camera_frame):
    """
    Creates the 'Big Screen + Small Feed' layout, optimized to fit any screen ratio.
    - Big Rectangle (Left): The AR Projector (Dynamically Scaled)
    - Small Rectangle (Right): The Live Camera Feed (Proportional Sidebar)
    """
    # 1. Define the Workspace Ratio
    # We use a 14:7 ratio base (1400x700) but calculate everything relatively
    base_h = 700
    proj_w = 1000
    sidebar_w = 400
    total_w = proj_w + sidebar_w

    # Create the master slate
    master_frame = np.zeros((base_h, total_w, 3), dtype=np.uint8)
    master_frame[:] = (18, 18, 18) # Dark aesthetic background

    # 2. Place the Projector Canvas (Left Side)
    # Ensure it fills exactly 1000x700 regardless of input size
    proj_resized = cv2.resize(projector_canvas, (proj_w, base_h))
    master_frame[0:base_h, 0:proj_w] = proj_resized

    # 3. Place the Live Camera Feed (Centered in Sidebar)
    # Calculate dimensions to ensure it never overflows the 400px sidebar
    cam_w, cam_h = 320, 240
    
    # Calculate X: Start at projector end (1000) + half of remaining sidebar space
    x_pos = proj_w + ((sidebar_w - cam_w) // 2)
    # Calculate Y: Center vertically in the 700px height
    y_pos = (base_h - cam_h) // 2
    
    cam_small = cv2.resize(live_camera_frame, (cam_w, cam_h))
    
    # Draw a tech-border around the camera feed
    # Border is slightly larger than the feed (2px padding)
    cv2.rectangle(master_frame, (x_pos-2, y_pos-2), (x_pos+cam_w+2, y_pos+cam_h+2), (0, 255, 0), 1)
    master_frame[y_pos:y_pos+cam_h, x_pos:x_pos+cam_w] = cam_small

    return master_frame