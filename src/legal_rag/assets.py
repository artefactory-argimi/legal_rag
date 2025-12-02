"""
Utilities for downloading and preparing encoder/index assets.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from sqlitedict import SqliteDict


def fetch_zip(uri: str, dest: Path) -> Path:
    """Download or copy a zip to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith(("http://", "https://")):
        urlretrieve(uri, dest)
        return dest
    src = Path(uri)
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
    index_zip_uri: str | Path,
    encoder_dest: Path,
    index_dest: Path,
) -> tuple[str, Path]:
    """Fetch and extract encoder and index assets; return encoder path and index dir.

    The index can be provided as a zipped archive (local path or remote URL) or as
    a pre-existing directory path that already contains the index files.
    """
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

    index_override = Path(index_zip_uri)
    index_ready = index_dest.exists() and any(index_dest.iterdir())
    if index_override.is_dir():
        index_dir = index_override
    elif index_ready:
        index_dir = index_dest
    else:
        index_dest.mkdir(parents=True, exist_ok=True)
        idx_zip_path = index_dest.parent / "_idx.zip"
        index_zip = fetch_zip(index_zip_uri, idx_zip_path)
        try:
            index_dir = extract_zip(index_zip, index_dest)
        finally:
            idx_zip_path.unlink(missing_ok=True)
    # Validate and normalize to the index root (folder passed to indexes.PLAID).
    fast_plaid_paths = list(index_dir.glob("**/fast_plaid_index"))
    if not fast_plaid_paths:
        raise ValueError(f"Invalid index layout at {index_dir}. Expected a fast_plaid_index directory under this folder.")
    fast_plaid = sorted(fast_plaid_paths, key=lambda p: len(p.relative_to(index_dir).parts))[0]
    # fast_plaid = <index_folder>/<index_name>/fast_plaid_index -> index_folder is parent of index_name.
    index_root = fast_plaid.parent.parent
    if not index_root.exists():
        raise ValueError(f"Could not resolve index root from fast_plaid_index at {fast_plaid}")
    index_dir = index_root

    return encoder_path, index_dir
