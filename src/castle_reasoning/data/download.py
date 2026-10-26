"""Data download."""

import hashlib
import json
import os
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path, PurePosixPath

import castle_reasoning.data.paths as paths

REQUIRED = {"plan.json", "model.mjb", "replay.npz"}


OPTIONAL = {"execution_config.json", "execution_order.json"}


def _record(entry) -> dict:
    lfs = getattr(entry, "lfs", None)
    digest = (
        lfs.get("sha256")
        if isinstance(lfs, dict)
        else getattr(lfs, "sha256", None)
    )
    if not digest:
        digest = getattr(entry, "blob_id", None)
    if not digest:
        raise ValueError(f"Missing authoritative HF file hash: {entry.path}")
    return {
        "path": entry.path,
        "size": entry.size,
        "source_hash": digest,
        "source_hash_kind": "sha256" if lfs else "git-blob-sha1",
    }


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected safe relative dataset path: {value}")
    return path


def _destination(root: Path, value: str) -> Path:
    path = root.joinpath(*_relative(value).parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Unsafe dataset path outside root: {value}")
    return path


def _verify(path: Path, record: dict) -> str:
    size = path.stat().st_size
    if size != record["size"]:
        raise ValueError(f"Dataset size mismatch: {record['path']}")
    sha256 = hashlib.sha256()
    git_hash = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha256.update(block)
            git_hash.update(block)
    actual = (
        sha256.hexdigest()
        if record["source_hash_kind"] == "sha256"
        else git_hash.hexdigest()
    )
    if actual != record["source_hash"]:
        raise ValueError(f"Dataset hash mismatch: {record['path']}")
    return sha256.hexdigest()


def _available(root, record, repo_id, revision):
    from huggingface_hub import hf_hub_download

    existing = _destination(root, record["path"])
    if existing.exists():
        _verify(existing, record)
        return existing
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=record["path"],
            token=False,
        )
    )
    _verify(cached, record)
    return cached


def _materialize(root, record, source):
    destination = _destination(root, record["path"])
    if destination.exists():
        return _verify(destination, record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        digest = _verify(temporary, record)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return _verify(destination, record)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _catalog(indices: dict[str, dict]) -> dict[str, tuple[str, dict]]:
    """Index accepted targets by campaign, retaining the first published demo."""
    catalog = {}
    for campaign, index in indices.items():
        for row in index["demonstrations"]:
            target_id = row["target_id"]
            if target_id in catalog and catalog[target_id][0] != campaign:
                raise ValueError(
                    f"Ambiguous target ID across HF campaigns: {target_id}"
                )
            catalog.setdefault(target_id, (campaign, row))
    return catalog


def download_tasks(
    task_ids: list[str],
    root: str | Path | None = None,
    *,
    revision: str | None = None,
    max_bytes: int = 256 * 1024 * 1024,
) -> dict:
    """Fetch existing attempts; never regenerate tasks or overwrite mismatches.

    HF Git blob IDs authenticate ordinary files and LFS SHA-256 authenticates
    binary files. The receipt additionally records SHA-256 for every local file.
    Metadata and size checks precede bulk transfer. Small index metadata may be
    fetched through the HF cache to select attempts before creating dataset/.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Install castle-reasoning-benchmark[download] for HF downloads"
        ) from error

    if not task_ids or max_bytes <= 0:
        raise ValueError("Task IDs and a positive transfer budget are required")
    specification = json.loads(
        files("castle_reasoning").joinpath("resources/dataset.json").read_text()
    )
    revision = revision or specification["revision"]
    api = HfApi(token=False)
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        revision = api.dataset_info(
            repo_id=specification["repo_id"], revision=revision, expand=["sha"]
        ).sha
    root = paths.dataset_root(root)
    manifest_path = root / "DATASET_MANIFEST.json"
    receipt = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "repo_id": specification["repo_id"],
            "revision": revision,
            "tasks": {},
            "files": {},
        }
    )
    if (
        receipt["revision"] != revision
        or receipt["repo_id"] != specification["repo_id"]
    ):
        raise ValueError(
            "Dataset revision/repository mismatch; use a different data root"
        )
    names = []
    for campaign in specification["campaigns"]:
        names.extend([f"{campaign}/index.json", f"{campaign}/campaign.json"])
    records = {
        _relative(entry.path).as_posix(): _record(entry)
        for entry in api.get_paths_info(
            repo_id=specification["repo_id"],
            paths=names,
            revision=revision,
            repo_type="dataset",
        )
    }
    if set(records) != set(names):
        raise ValueError("Required HF index/campaign metadata is missing")
    if sum(row["size"] for row in records.values()) > max_bytes:
        raise ValueError("Dataset metadata exceeds transfer budget")
    metadata_sources = {
        name: _available(root, record, specification["repo_id"], revision)
        for name, record in records.items()
    }
    indices = {
        campaign: json.loads(
            metadata_sources[f"{campaign}/index.json"].read_text()
        )
        for campaign in specification["campaigns"]
    }
    catalog = _catalog(indices)
    selected = {}
    for task_id in dict.fromkeys(task_ids):
        if task_id not in catalog:
            raise ValueError(f"No accepted HF attempt for {task_id}")
        campaign, row = catalog[task_id]
        attempt = f"{campaign}/{_relative(row['attempt_directory']).as_posix()}"
        entries = list(
            api.list_repo_tree(
                repo_id=specification["repo_id"],
                path_in_repo=attempt,
                revision=revision,
                repo_type="dataset",
                recursive=False,
            )
        )
        task_files = {
            entry.path: _record(entry)
            for entry in entries
            if PurePosixPath(entry.path).parent.as_posix() == attempt
            and PurePosixPath(entry.path).name in REQUIRED | OPTIONAL
        }
        missing = REQUIRED - {PurePosixPath(name).name for name in task_files}
        if missing:
            raise ValueError(
                f"Missing required files for {task_id}: {sorted(missing)}"
            )
        records.update(task_files)
        selected[task_id] = row["attempt_directory"]
    requested = sum(row["size"] for row in records.values())
    if requested > max_bytes:
        raise ValueError(
            f"Selected dataset needs {requested} bytes, exceeding transfer budget {max_bytes}"
        )
    root.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        source = metadata_sources.get(name)
        if source is None:
            source = _available(root, record, specification["repo_id"], revision)
        digest = _materialize(root, record, source)
        receipt["files"][name] = {**record, "sha256": digest}
    receipt["tasks"].update(selected)
    receipt["task_campaigns"] = {
        task_id: catalog[task_id][0] for task_id in receipt["tasks"]
    }
    receipt["selected_bytes"] = sum(
        row["size"] for row in receipt["files"].values()
    )
    with tempfile.NamedTemporaryFile(
        mode="w", dir=root, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    temporary.replace(manifest_path)
    return receipt
