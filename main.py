import sys
import ctypes
import threading
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from interface.ui_hub import UIHub
from core.main_kernel import AIILAKernel
import multiprocessing as mp

def launch():
    # --- PYQT6 DPI CONFIGURATION ---
    # We let PyQt6 handle DPI context natively to avoid "Access Denied" errors.
    # This ensures your 1920x1080 projector window maps to physical pixels correctly.
    if sys.platform == 'win32':
        try:
            # Setting HighDPI scaling attributes before creating the Application
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    # --- PYQT6 APPLICATION INITIALIZATION ---
    app = QApplication(sys.argv)
    
    # Initialize Kernel Logic
    kernel = AIILAKernel()
    
    # Initialize PyQt6 UI and link to Kernel
    # The UIHub now manages its own high-speed QTimer for polling
    window = UIHub(kernel)
    
    # Linking the Kernel callback to the UI bridge
    kernel.gui_callback = window.update_display
    
    # Display the Main Dashboard
    window.show()

    # --- KERNEL BACKGROUND THREAD ---
    # Separation of tracking logic and UI rendering is vital for responsiveness
    thread = threading.Thread(target=kernel.run, daemon=True)
    thread.start()
    
    # Start the hardware-accelerated event loop
    sys.exit(app.exec())

# --- CRITICAL MULTIPROCESSING SAFETY GUARD ---
# Mandatory on Windows when using mp.Process to avoid infinite recursion.
if __name__ == "__main__":
    # Required for Windows multiprocessing support
    mp.freeze_support() 
    launch()