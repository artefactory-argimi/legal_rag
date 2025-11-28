"""
Utilities for downloading and preparing encoder/index assets.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def fetch_zip(uri: str, dest: Path) -> Path:
    """Download or copy a zip to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith(("http://", "https://")):
        urlretrieve(uri, dest)
        return dest
    src = Path(uri)
    if not src.exists():
        raise FileNotFoundError(f"Zip file not found: {src}")
    shutil.copy(src, dest)
    return dest


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extract a zip to target_dir (cleaning previous contents)."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return target_dir


def resolve_model_dir(base_dir: Path) -> Path:
    """Return a directory that contains a model config; prefer base, else a child."""
    if (base_dir / "config.json").exists() or (base_dir / "config_sentence_transformers.json").exists():
        return base_dir
    for cand in base_dir.iterdir():
        if cand.is_dir() and (
            (cand / "config.json").exists() or (cand / "config_sentence_transformers.json").exists()
        ):
            return cand
    return base_dir


def prepare_assets(
    encoder_zip_uri: str,
    index_zip_uri: str,
    encoder_dest: Path,
    index_dest: Path,
) -> tuple[str, Path]:
    """Fetch and extract encoder and index assets; return encoder path and index dir."""
    encoder_ready = encoder_dest.exists() and any(encoder_dest.iterdir())
    if encoder_ready:
        encoder_dir = resolve_model_dir(encoder_dest)
    else:
        encoder_dest.mkdir(parents=True, exist_ok=True)
        enc_zip_path = encoder_dest.parent / "_enc.zip"
        encoder_zip = fetch_zip(encoder_zip_uri, enc_zip_path)
        try:
            encoder_dir = extract_zip(encoder_zip, encoder_dest)
            encoder_dir = resolve_model_dir(encoder_dir)
        finally:
            enc_zip_path.unlink(missing_ok=True)
    encoder_path = str(encoder_dir.resolve())

    index_ready = index_dest.exists() and any(index_dest.iterdir())
    if index_ready:
        index_dir = index_dest
    else:
        index_dest.mkdir(parents=True, exist_ok=True)
        idx_zip_path = index_dest.parent / "_idx.zip"
        index_zip = fetch_zip(index_zip_uri, idx_zip_path)
        try:
            index_dir = extract_zip(index_zip, index_dest)
        finally:
            idx_zip_path.unlink(missing_ok=True)
    # Some archives contain an extra leading "index/" folder; reuse it if present.
    nested_index_dir = index_dir / "index"
    if nested_index_dir.is_dir():
        index_dir = nested_index_dir

    return encoder_path, index_dir
