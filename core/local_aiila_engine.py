from __future__ import annotations

import re


def _compact(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalized(text: str) -> str:
    text = _compact(text).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def _bool_text(value: bool, on: str = "on", off: str = "off") -> str:
    return on if value else off


def _status_parts(app_state: dict) -> list[str]:
    model = app_state.get("active_model") or "no scanned unit"
    ar_mode = (app_state.get("ar_mode") or "default").upper()
    layer = app_state.get("current_layer_view", 1)
    circuit = _bool_text(bool(app_state.get("circuit_engine_enabled")))
    draw_mode = "on" if app_state.get("ar_mode") == "draw" else "off"
    simulation = _bool_text(bool(app_state.get("simulation_running")))
    projector = _bool_text(bool(app_state.get("projector_enabled")))
    calibration = _bool_text(bool(app_state.get("calibration_mode")))
    parts = [
        f"target {model}",
        f"AR mode {ar_mode}",
        f"layer {layer}",
        f"circuit {circuit}",
        f"wire draw {draw_mode}",
        f"simulation {simulation}",
        f"projector {projector}",
        f"calibration {calibration}",
    ]
    if app_state.get("exploded_view_visible"):
        parts.append(
            f"internal view part {app_state.get('exploded_view_index', 0)} of "
            f"{app_state.get('exploded_view_total', 0)}"
        )
    return parts


def generate_offline_response(app_state: dict, user_text: str) -> str:
    text = _normalized(user_text)
    if not text:
        return "AIILA offline voice is ready."

    if any(word in text for word in ("hello", "hi aiila", "hey aiila", "wake up")):
        return "AIILA offline and ready."

    if "help" in text or "what can you do" in text or "commands" in text:
        return (
            "Offline AIILA can report status, scanned device, layer, AR mode, "
            "circuit state, simulation state, projector state, selected tool, "
            "internal-view progress, and it supports local voice actions like scan, "
            "page change, simulation control, projector control, calibration, save, undo, and settings."
        )

    if "status" in text or "system report" in text or "report" in text:
        return "Status: " + ". ".join(_status_parts(app_state)) + "."

    if any(phrase in text for phrase in (
        "what is scanned",
        "what did you scan",
        "what phone is this",
        "what device is this",
        "what object is this",
        "what is the model",
        "current target",
    )):
        model = app_state.get("active_model")
        category = app_state.get("active_category")
        if not model or model in {"unknown", "error"}:
            return "No reliable scan is active yet. Use scan unit first."
        if category:
            return f"Current target is {model} in category {category}."
        return f"Current target is {model}."

    if any(phrase in text for phrase in ("layer 2", "second layer", "go to layer two", "switch to layer two")):
        app_state["current_layer_view"] = 2
        return "Layer 2 selected."

    if any(phrase in text for phrase in ("layer 1", "first layer", "go to layer one", "switch to layer one")):
        app_state["current_layer_view"] = 1
        return "Layer 1 selected."

    if "current layer" in text or "which layer" in text:
        return f"Current layer is {app_state.get('current_layer_view', 1)}."

    if "current page" in text or "which page" in text:
        return f"Current page is {app_state.get('current_layer_view', 1)}."

    if "ar mode" in text or "current mode" in text or "which mode" in text:
        mode = (app_state.get("ar_mode") or "default").upper()
        return f"Current AR mode is {mode}."

    if "circuit mode" in text:
        enabled = bool(app_state.get("circuit_engine_enabled"))
        return f"Circuit mode is {_bool_text(enabled, 'enabled', 'disabled')}."

    if "wire draw" in text or "draw mode" in text:
        enabled = app_state.get("ar_mode") == "draw"
        return f"Wire draw mode is {_bool_text(enabled, 'enabled', 'disabled')}."

    if "simulation" in text:
        enabled = bool(app_state.get("simulation_running"))
        return f"Simulation is {_bool_text(enabled, 'running', 'stopped')}."

    if "projector" in text or "screen cast" in text:
        enabled = bool(app_state.get("projector_enabled"))
        return f"Projector is {_bool_text(enabled, 'active', 'inactive')}."

    if "calibration" in text:
        enabled = bool(app_state.get("calibration_mode"))
        return f"Calibration grid is {_bool_text(enabled, 'enabled', 'disabled')}."

    if "selected tool" in text or "current tool" in text or "which tool" in text:
        tool = (app_state.get("selected_tool") or "unknown").replace("_", " ")
        return f"Selected tool is {tool}."

    if "internal view" in text or "exploded view" in text or "inside view" in text:
        if app_state.get("exploded_view_visible"):
            caption = app_state.get("exploded_view_caption") or "internal image"
            idx = app_state.get("exploded_view_index", 0)
            total = app_state.get("exploded_view_total", 0)
            return f"Internal view is open on part {idx} of {total}. {caption}."
        if app_state.get("active_model"):
            return "Internal view is closed. Say show exploded view to open it."
        return "Scan a device first, then ask for the exploded view."

    recent_feedback = app_state.get("feedback_log") or []
    if recent_feedback:
        try:
            _ts, last_msg = recent_feedback[-1]
            return f"Last kernel update: {last_msg}."
        except Exception:
            pass

    return (
        f"Offline AIILA heard: {user_text}. "
        "Ask for status, scanned device, current mode, layer, or internal view."
    )
