"""
modeling_engine.py  ·  AIILA 3D Engine (Standalone Module)
=========================================================
Enhanced 3D wireframe engine for AR projection. 
Features: Scene Graph, Procedural Primitives, and Axis-specific Rotations.
"""

import cv2
import numpy as np
import math

class ModelingEngine:
    def __init__(self, canvas_w=1000, canvas_h=700):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        
        # Virtual Camera Intrinsics
        # f = focal length. Higher f = narrow field of view (more zoom).
        self.f = 800 
        self.cx = canvas_w // 2
        self.cy = canvas_h // 2
        
        # Scene Graph: List of 3D Objects
        self.objects = []
        self._init_default_scene()

    def _init_default_scene(self):
        """Initializes the engine with default 3D geometry."""
        # Add a Cube at the center
        self.add_primitive("cube", pos=[0, 0, 600], scale=60)
        # Add a Pyramid to the left
        self.add_primitive("pyramid", pos=[-150, 0, 700], scale=50)

    def add_primitive(self, kind: str, pos: list, scale: float = 50.0):
        """Generates procedural geometry based on type."""
        if kind == "cube":
            # 8 Vertices
            verts = np.array([
                [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                [-1,-1,1],  [1,-1,1],  [1,1,1],  [-1,1,1]
            ], dtype=np.float32) * scale
            # 12 Edges
            edges = [
                (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), 
                (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)
            ]
        elif kind == "pyramid":
            # 5 Vertices
            verts = np.array([
                [0,-1,0], [-1,1,-1], [1,1,-1], [1,1,1], [-1,1,1]
            ], dtype=np.float32) * scale
            # 8 Edges
            edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
        else:
            return

        self.objects.append({
            "type": kind,
            "verts": verts,
            "edges": edges,
            "pos": np.array(pos, dtype=np.float32),
            "rot": np.array([0, 0, 0], dtype=np.float32), # Euler angles (X, Y, Z)
            "color": (0, 255, 255) # Default Cyan
        })

    def transform_object(self, index: int, translate=None, rotate=None):
        """Applies translation or rotation to a specific object in the scene."""
        if 0 <= index < len(self.objects):
            obj = self.objects[index]
            if translate:
                obj["pos"] += np.array(translate, dtype=np.float32)
            if rotate:
                obj["rot"] += np.array(rotate, dtype=np.float32)

    def _get_rotation_matrix(self, rx, ry, rz):
        """Calculates a combined 3D rotation matrix."""
        # Convert degrees to radians
        ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
        
        # Rotation X
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(ax), -math.sin(ax)],
            [0, math.sin(ax), math.cos(ax)]
        ])
        # Rotation Y
        Ry = np.array([
            [math.cos(ay), 0, math.sin(ay)],
            [0, 1, 0],
            [-math.sin(ay), 0, math.cos(ay)]
        ])
        # Rotation Z
        Rz = np.array([
            [math.cos(az), -math.sin(az), 0],
            [math.sin(az), math.cos(az), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    def render(self, canvas: np.ndarray):
        """Projects and renders all scene objects onto the AR canvas."""
        for obj in self.objects:
            # 1. Generate Rotation Matrix for this frame
            R = self._get_rotation_matrix(*obj["rot"])
            
            # 2. Transform Vertices: Rotate then Translate
            # Using NumPy broadcasting for speed
            transformed_verts = (obj["verts"] @ R.T) + obj["pos"]
            
            # 3. Project 3D to 2D
            projected_2d = []
            for v in transformed_verts:
                # Basic Pinhole Projection: x' = (x*f)/z + cx
                z = v[2] if v[2] != 0 else 1 # Avoid division by zero
                px = int((v[0] * self.f) / z + self.cx)
                py = int((v[1] * self.f) / z + self.cy)
                projected_2d.append((px, py))

            # 4. Draw Edges on OpenCV Canvas
            for edge in obj["edges"]:
                p1, p2 = projected_2d[edge[0]], projected_2d[edge[1]]
                # Only draw if points are likely on screen
                cv2.line(canvas, p1, p2, obj["color"], 2, cv2.LINE_AA)

            # 5. Draw Vertex Points (Optional aesthetic choice)
            for p in projected_2d:
                cv2.circle(canvas, p, 3, (255, 255, 255), -1)

    def clear_scene(self):
        """Removes all objects from the engine."""
        self.objects = []