import sys
import signal
import ctypes
import threading
import traceback
import multiprocessing as mp

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer

from interface.ui_hub import UIHub
from core.main_kernel import AIILAKernel


def launch():
    shutting_down = {'value': False}

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
        if exc_type is KeyboardInterrupt:
            return
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

    def _shutdown():
        if shutting_down['value']:
            return
        shutting_down['value'] = True
        window.shutdown()
        app.quit()

    sigint_pump = QTimer()
    sigint_pump.setInterval(200)
    sigint_pump.timeout.connect(lambda: None)
    sigint_pump.start()

    def _handle_sigint(_signum, _frame):
        QTimer.singleShot(0, _shutdown)

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        signal.signal(signal.SIGTERM, _handle_sigint)
    except Exception:
        pass

    app.aboutToQuit.connect(_shutdown)

    exit_code = app.exec()
    window.shutdown()
    thread.join(timeout=2.0)
    sys.exit(exit_code)


def _run_kernel_safe(kernel: AIILAKernel):
    """Wraps kernel.run() so a crash in the background thread is printed clearly."""
    try:
        kernel.run()
    except KeyboardInterrupt:
        kernel.running = False
    except RuntimeError as exc:
        if (not kernel.running
                and "cannot schedule new futures after shutdown" in str(exc)):
            return
        traceback.print_exc()
        kernel.running = False
    except Exception:
        traceback.print_exc()
        kernel.running = False   # stop the loop cleanly on error


if __name__ == "__main__":
    mp.freeze_support()
    launch()
