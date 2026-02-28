import sys
import ctypes
import threading
import traceback
import multiprocessing as mp

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from interface.ui_hub import UIHub
from core.main_kernel import AIILAKernel


def launch():

    # ── DPI awareness (Windows) ───────────────────────────────────────────────
    if sys.platform == 'win32':
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    # ── Application ──────────────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setStyle('Fusion')   # consistent cross-platform look

    # ── Global exception handler — shows a dialog instead of silent crash ────
    def _handle_exception(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(msg, file=sys.stderr)
        try:
            dlg = QMessageBox()
            dlg.setWindowTitle("AIILA — Unhandled Error")
            dlg.setIcon(QMessageBox.Icon.Critical)
            dlg.setText(str(exc_value))
            dlg.setDetailedText(msg)
            dlg.exec()
        except Exception:
            pass   # if Qt itself is broken, just print to stderr
    sys.excepthook = _handle_exception

    # ── Kernel ────────────────────────────────────────────────────────────────
    kernel = AIILAKernel()

    # ── UI ────────────────────────────────────────────────────────────────────
    window = UIHub(kernel)
    kernel.gui_callback = window.update_display
    window.show()

    # ── Kernel thread ─────────────────────────────────────────────────────────
    thread = threading.Thread(target=_run_kernel_safe, args=(kernel,), daemon=True)
    thread.start()

    sys.exit(app.exec())


def _run_kernel_safe(kernel: AIILAKernel):
    """Wraps kernel.run() so a crash in the background thread is printed clearly."""
    try:
        kernel.run()
    except Exception:
        traceback.print_exc()
        kernel.running = False   # stop the loop cleanly on error


if __name__ == "__main__":
    mp.freeze_support()
    launch()