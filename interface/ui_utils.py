import numpy as np
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect

from interface.ui_styles import C

def _ndarray_to_pixmap(rgb: np.ndarray) -> QPixmap:
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())

def _glow(widget, color: str = C['accent'], radius: int = 18):
    fx = QGraphicsDropShadowEffect(widget)
    fx.setBlurRadius(radius)
    fx.setOffset(0, 0)
    fx.setColor(QColor(color))
    widget.setGraphicsEffect(fx)
