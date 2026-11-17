"""Capture the source and configuration used by a local experiment."""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess

from cardioagent.common.io import digest, save_json


def json_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def experiment_provenance(config: dict) -> dict:
    package = Path(__file__).resolve().parents[2]
    root = package.parent.parent
    files = set(package.rglob("*.py"))
    for directory in ("tests", "scripts", "configs", "config"):
        files.update(path for path in (root / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("."))
    files.update(path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file())
    hashes = {path.relative_to(root).as_posix(): digest(path) for path in sorted(files)}
    try:
        git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True, timeout=5)
        commit = git.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    versions = {}
    for package_name in ("numpy", "scipy", "wfdb", "torch", "torchaudio", "langchain", "langgraph",
                         "langchain-openai", "matplotlib"):
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return {
        "git_commit": commit, "python": platform.python_version(), "packages": versions,
        "source_file_sha256": hashes, "source_tree_sha256": json_hash(hashes),
        "configuration": config, "configuration_sha256": json_hash(config),
    }


def save_provenance(output: Path, config: dict) -> dict:
    manifest = experiment_provenance(config)
    save_json(output / "source_manifest.json", manifest)
    return {key: manifest[key] for key in ("source_tree_sha256", "configuration_sha256", "git_commit")}
