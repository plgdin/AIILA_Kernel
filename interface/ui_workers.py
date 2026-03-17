import cv2
import multiprocessing as mp

from interface.ar_overlay import assemble_final_os_view

def _image_worker(in_q: mp.Queue, out_q: mp.Queue):
    while True:
        try:
            ar_canvas, raw_frame, state, proj_active, calib = in_q.get()

            p_rgb = None
            if proj_active:
                p_rgb = cv2.cvtColor(ar_canvas, cv2.COLOR_BGR2RGB)
                if calib:
                    h, w = p_rgb.shape[:2]
                    for x in range(0, w, 100):
                        cv2.line(p_rgb, (x, 0), (x, h), (0, 255, 0), 1)
                    for y in range(0, h, 100):
                        cv2.line(p_rgb, (0, y), (w, y), (0, 255, 0), 1)

            combined = assemble_final_os_view(ar_canvas, raw_frame)
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            ar_rgb = cv2.resize(combined_rgb, (1100, 620), interpolation=cv2.INTER_LINEAR)

            cam_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            cam_rgb = cv2.resize(cam_rgb, (280, 190), interpolation=cv2.INTER_LINEAR)

            while not out_q.empty():
                try:
                    out_q.get_nowait()
                except Exception:
                    pass
            out_q.put((p_rgb, ar_rgb, cam_rgb, state))
        except Exception:
            continue
