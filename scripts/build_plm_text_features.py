import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from v1_dataset import artifact_stem


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INPUT_ROOT = Path("data/processed/v1_inputs")
DEFAULT_OUTPUT_ROOT = Path("data/processed/v1_text_features")
SUPPORTED_DATASETS = [
    "pheme",
    "twitter15",
    "twitter16",
    "twitter15_rumdetect2017",
    "twitter16_rumdetect2017",
    "weibo",
]


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def resolve_model_source(model_name: str, local_files_only: bool) -> str:
    if not local_files_only:
        return model_name
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_cache = cache_home / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    snapshots = sorted(
        [path for path in model_cache.glob("*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        raise FileNotFoundError(
            f"No local HuggingFace snapshot found for {model_name}. "
            "Run once without --local-files-only to download it."
        )
    return str(snapshots[0])


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return functional.normalize(pooled, p=2, dim=1)


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    parts = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch_texts = texts[start : start + batch_size]
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            output = model(**tokens)
            embeddings = mean_pool(output.last_hidden_state, tokens["attention_mask"])
            parts.append(embeddings.cpu().numpy().astype(np.float32))
            if batch_number == 1 or batch_number % 20 == 0 or batch_number == total_batches:
                print(
                    f"encoded {min(start + len(batch_texts), len(texts))}/{len(texts)} "
                    f"({batch_number}/{total_batches} batches)",
                    flush=True,
                )
    return np.concatenate(parts, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pheme", choices=SUPPORTED_DATASETS)
    parser.add_argument("--observation", type=int, default=180)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--feature-name", default="multilingual_minilm")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    torch.set_num_threads(max(args.torch_threads, 1))
    stem = artifact_stem(args.dataset, args.observation)
    input_path = Path(args.input_root) / args.dataset / f"{stem}.npz"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir = Path(args.output_root) / args.dataset
    suffix = "" if args.max_samples <= 0 else f"_first{args.max_samples}"
    output_path = output_dir / f"{args.feature_name}{suffix}.npz"
    metadata_path = output_dir / f"{args.feature_name}{suffix}_metadata.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists; use --force to rebuild")

    archive = np.load(input_path, allow_pickle=False)
    try:
        sample_ids = archive["sample_ids"].astype(np.str_)
        source_texts = archive["source_texts"].astype(np.str_)
    finally:
        archive.close()
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]
        source_texts = source_texts[: args.max_samples]
    texts = [str(text).strip() for text in source_texts]
    if not texts or any(not text for text in texts):
        raise ValueError("PLM text features require non-empty source text for every selected sample")

    started = time.perf_counter()
    model_source = resolve_model_source(args.model_name, args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    features = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    if not np.isfinite(features).all():
        raise ValueError("PLM encoder produced non-finite features")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_ids=sample_ids,
        text_features=features,
    )
    metadata = {
        "dataset": args.dataset,
        "source_artifact": str(input_path),
        "feature_name": args.feature_name,
        "model_name": args.model_name,
        "model_source": model_source,
        "model_slug": model_slug(args.model_name),
        "pooling": "attention_mask_mean_l2_normalized",
        "max_length": args.max_length,
        "num_samples": int(len(sample_ids)),
        "feature_dim": int(features.shape[1]),
        "device": str(device),
        "local_files_only": bool(args.local_files_only),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "leakage_rule": "source post text only; no reaction or future text",
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "metadata": str(metadata_path),
                "shape": list(features.shape),
                "elapsed_seconds": metadata["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
