"""
ui_styles.py  ·  AIILA OS Styling Configuration
=============================================
Contains all colors, fonts, and base stylesheets used across the UI.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    'bg':          '#020408',
    'surface':     '#060d14',
    'panel':       '#0a1520',
    'border':      '#0e2030',
    'border_hi':   '#1a3f5c',
    'accent':      '#00c8ff',
    'accent2':     '#ff4d1a',
    'accent3':     '#00ff88',
    'text':        '#8ab0c8',
    'text_bright': '#d4eaf8',
    'text_dim':    '#2a4a60',
    'danger':      '#ff2244',
    'warn':        '#ffaa00',
}

MONO  = "Courier New"
TITLE = "Courier New"

_SS_BASE = f"""
    QWidget {{
        background: {C['bg']};
        color: {C['text_bright']};
        font-family: '{MONO}';
        font-size: 12px;
    }}
    QScrollBar:vertical {{
        background: {C['surface']};
        width: 6px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {C['border_hi']};
        border-radius: 3px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""
