"""3D mesh import and processing for aero visualization."""

from __future__ import annotations

import struct
import time
import uuid
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".stl", ".obj"}


def _parse_ascii_stl(content: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ASCII STL into vertices, face indices, and face normals."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    normals: list[list[float]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}

    lines = content.splitlines()
    i = 0
    current_normal = [0.0, 0.0, 1.0]
    face_indices: list[int] = []

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("facet normal"):
            parts = line.split()
            current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
        elif line.startswith("vertex"):
            parts = line.split()
            v = (float(parts[1]), float(parts[2]), float(parts[3]))
            if v not in vertex_map:
                vertex_map[v] = len(vertices)
                vertices.append(list(v))
            face_indices.append(vertex_map[v])
        elif line.startswith("endfacet") and len(face_indices) == 3:
            faces.append(face_indices)
            normals.append(current_normal)
            face_indices = []
        i += 1

    return np.array(vertices), np.array(faces), np.array(normals)


def _parse_binary_stl(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse binary STL."""
    if len(data) < 84:
        raise ValueError("Invalid binary STL file")

    n_triangles = struct.unpack("<I", data[80:84])[0]
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    normals: list[list[float]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}

    offset = 84
    for _ in range(n_triangles):
        nx, ny, nz = struct.unpack("<3f", data[offset : offset + 12])
        offset += 12
        face_indices: list[int] = []
        for _ in range(3):
            x, y, z = struct.unpack("<3f", data[offset : offset + 12])
            offset += 12
            v = (round(x, 6), round(y, 6), round(z, 6))
            if v not in vertex_map:
                vertex_map[v] = len(vertices)
                vertices.append([x, y, z])
            face_indices.append(vertex_map[v])
        offset += 2  # attribute byte count
        faces.append(face_indices)
        normals.append([nx, ny, nz])

    return np.array(vertices), np.array(faces), np.array(normals)


def parse_stl(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Auto-detect ASCII or binary STL."""
    if data[:5].lower() == b"solid":
        try:
            text = data.decode("utf-8", errors="ignore")
            if "facet normal" in text:
                return _parse_ascii_stl(text)
        except UnicodeDecodeError:
            pass
    return _parse_binary_stl(data)


def parse_obj(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse OBJ file (vertices + faces, triangulated)."""
    text = data.decode("utf-8", errors="ignore")
    raw_vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            raw_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            face = []
            for p in parts[1:4]:
                idx = int(p.split("/")[0]) - 1
                face.append(idx)
            faces.append(face)

    vertices = np.array(raw_vertices)
    faces_arr = np.array(faces)

    normals = []
    for face in faces:
        v0, v1, v2 = raw_vertices[face[0]], raw_vertices[face[1]], raw_vertices[face[2]]
        e1 = np.array(v1) - np.array(v0)
        e2 = np.array(v2) - np.array(v0)
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        normals.append((n / norm).tolist() if norm > 0 else [0, 0, 1])

    return vertices, faces_arr, np.array(normals)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute smooth vertex normals from face data."""
    vnorms = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        e1 = v1 - v0
        e2 = v2 - v0
        fn = np.cross(e1, e2)
        norm = np.linalg.norm(fn)
        if norm > 0:
            fn /= norm
        for idx in face:
            vnorms[idx] += fn

    for i in range(len(vnorms)):
        n = np.linalg.norm(vnorms[i])
        if n > 0:
            vnorms[i] /= n
        else:
            vnorms[i] = [0, 0, 1]

    return vnorms


def cleanup_uploads(max_files: int = 10, max_age_seconds: int = 7200) -> int:
    """Clean up old uploaded models in MODELS_DIR to prevent disk bloat (Bug B7)."""
    now = time.time()
    deleted = 0
    files = [
        p for p in MODELS_DIR.iterdir()
        if p.is_file() and p.name != ".gitkeep" and not p.name.startswith(".")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for i, f in enumerate(files):
        age = now - f.stat().st_mtime
        if i >= max_files or age > max_age_seconds:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
    return deleted


def save_upload(filename: str, data: bytes) -> dict:
    """Save uploaded mesh and return metadata."""
    cleanup_uploads()
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Only .stl and .obj are supported.")

    if ext == ".stl":
        vertices, faces, _ = parse_stl(data)
    else:
        vertices, faces, _ = parse_obj(data)

    if len(vertices) == 0:
        raise ValueError("Mesh contains no vertices.")

    vnorms = vertex_normals(vertices, faces)
    model_id = str(uuid.uuid4())[:8]
    dest = MODELS_DIR / f"{model_id}{ext}"
    dest.write_bytes(data)

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)

    return {
        "model_id": model_id,
        "filename": filename,
        "format": ext.lstrip("."),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "bounds": {"min": vmin.tolist(), "max": vmax.tolist()},
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "normals": vnorms.tolist(),
    }


def get_model_path(model_id: str) -> Path | None:
    """Find saved model file by ID."""
    for p in MODELS_DIR.glob(f"{model_id}.*"):
        return p
    return None
