#!/usr/bin/env python3
"""Offline KMeans clustering for item-intent embeddings.

The default KMeans parameters follow IACLR's semantic-intent initialization:
``KMeans(n_clusters=K, random_state=2024, n_init=10)``.  The exported cluster
centers can be loaded as fixed item-intent prototypes in later training runs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster item-intent embeddings and export intent prototypes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input embedding file (.pkl or .npy), shaped [num_items, dim].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npy file for cluster centers, shaped [num_clusters, dim].",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=128,
        help="Number of item-intent prototypes (default: 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="KMeans random seed (default: 2024, identical to IACLR).",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of KMeans initializations (default: 10, identical to IACLR).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum iterations for each initialization (default: 300).",
    )
    parser.add_argument(
        "--method",
        choices=("kmeans", "minibatch"),
        default="kmeans",
        help="Exact IACLR-style KMeans or lower-memory MiniBatchKMeans.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size used only by --method minibatch (default: 1024).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize item vectors before clustering. Disabled to match IACLR.",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=None,
        help="Optional .npy output for each item's cluster id.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional JSON output. Defaults to <output>.json.",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as file:
            raw: Any = pickle.load(file)
    elif suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    else:
        raise ValueError(
            f"Unsupported input format {suffix!r}; expected .pkl, .pickle, or .npy"
        )

    embeddings = np.asarray(raw, dtype=np.float32, order="C")
    del raw
    if embeddings.ndim != 2:
        raise ValueError(
            "Item-intent embeddings must be a 2-D matrix shaped "
            f"[num_items, dim], got {embeddings.shape}"
        )
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"Item-intent embedding matrix is empty: {embeddings.shape}")

    finite_mask = np.isfinite(embeddings)
    if not finite_mask.all():
        invalid_count = int(embeddings.size - finite_mask.sum())
        raise ValueError(f"Found {invalid_count} NaN/Inf values in {path}")
    return embeddings


def validate_output_paths(args: argparse.Namespace) -> Path:
    metadata_path = args.metadata_output or args.output.with_suffix(".json")
    if args.output.suffix.lower() != ".npy":
        raise ValueError(f"--output must end in .npy: {args.output}")
    if args.labels_output is not None and args.labels_output.suffix.lower() != ".npy":
        raise ValueError(f"--labels-output must end in .npy: {args.labels_output}")
    if metadata_path.suffix.lower() != ".json":
        raise ValueError(f"--metadata-output must end in .json: {metadata_path}")

    named_paths = {
        "input": args.input.resolve(),
        "output": args.output.resolve(),
        "metadata-output": metadata_path.resolve(),
    }
    if args.labels_output is not None:
        named_paths["labels-output"] = args.labels_output.resolve()
    if len(set(named_paths.values())) != len(named_paths):
        details = ", ".join(f"{name}={path}" for name, path in named_paths.items())
        raise ValueError(f"Input and output paths must be distinct; got {details}")
    return metadata_path


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] == 0)
    if zero_rows.size:
        preview = ", ".join(map(str, zero_rows[:10]))
        raise ValueError(
            f"Cannot L2-normalize {zero_rows.size} zero vectors; first rows: {preview}"
        )
    return np.ascontiguousarray(embeddings / norms, dtype=np.float32)


def fit_kmeans(
    embeddings: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, float, int]:
    if args.num_clusters <= 0:
        raise ValueError("--num-clusters must be positive")
    if args.num_clusters > embeddings.shape[0]:
        raise ValueError(
            f"--num-clusters ({args.num_clusters}) exceeds the number of items "
            f"({embeddings.shape[0]})"
        )
    if args.n_init <= 0 or args.max_iter <= 0:
        raise ValueError("--n-init and --max-iter must be positive")

    common_args = dict(
        n_clusters=args.num_clusters,
        random_state=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    if args.method == "kmeans":
        clusterer = KMeans(**common_args)
    else:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        clusterer = MiniBatchKMeans(
            **common_args,
            batch_size=args.batch_size,
            reassignment_ratio=0.01,
        )

    labels = clusterer.fit_predict(embeddings)
    centers = np.asarray(clusterer.cluster_centers_, dtype=np.float32, order="C")
    return (
        centers,
        labels.astype(np.int32, copy=False),
        float(clusterer.inertia_),
        int(clusterer.n_iter_),
    )


def save_array(path: Path, array: np.ndarray) -> None:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Output path must end in .npy: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    metadata_path = validate_output_paths(args)

    embeddings = load_embeddings(args.input)
    if args.normalize:
        embeddings = l2_normalize(embeddings)

    print(
        f"Loaded {embeddings.shape[0]} item intents with dimension "
        f"{embeddings.shape[1]} from {args.input}"
    )
    print(
        f"Running {args.method}: clusters={args.num_clusters}, seed={args.seed}, "
        f"n_init={args.n_init}, normalize={args.normalize}"
    )

    centers, labels, inertia, n_iter = fit_kmeans(embeddings, args)
    cluster_sizes = np.bincount(labels, minlength=args.num_clusters)
    save_array(args.output, centers)

    if args.labels_output is not None:
        save_array(args.labels_output, labels)

    elapsed = time.perf_counter() - started_at
    metadata = {
        "input": str(args.input.resolve()),
        "prototypes": str(args.output.resolve()),
        "labels": str(args.labels_output.resolve()) if args.labels_output else None,
        "input_shape": list(embeddings.shape),
        "prototype_shape": list(centers.shape),
        "dtype": str(centers.dtype),
        "method": args.method,
        "normalized_before_clustering": args.normalize,
        "num_clusters": args.num_clusters,
        "seed": args.seed,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "n_iter": n_iter,
        "inertia": inertia,
        "cluster_sizes": cluster_sizes.tolist(),
        "elapsed_seconds": elapsed,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Saved prototypes {centers.shape} ({centers.dtype}) to {args.output}")
    if args.labels_output is not None:
        print(f"Saved item cluster ids {labels.shape} to {args.labels_output}")
    print(f"Saved clustering metadata to {metadata_path}")
    print(
        f"Cluster size min/mean/max: {cluster_sizes.min()}/"
        f"{cluster_sizes.mean():.2f}/{cluster_sizes.max()}, "
        f"inertia={inertia:.6f}, elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
