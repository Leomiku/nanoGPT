# ./hfd.sh HuggingFaceFW/fineweb-edu --dataset --tool aria2c -x 4 --include "sample/100BT/*" --local-dir /root/autodl-tmp/cache/fineweb-edu

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


DEFAULT_DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "Sample-100BT"
DEFAULT_TRAIN_TOKENS = 20_000_000_000
DEFAULT_VAL_TOKENS = 20_000_000
DEFAULT_VAL_RATIO = 0.001
DEFAULT_SHUFFLE_BUFFER_SIZE = 10_000
PADDED_GPT2_VOCAB_SIZE = 50304
UINT16_BYTES = np.dtype(np.uint16).itemsize


def parse_args():
	parser = argparse.ArgumentParser(
		description="Stream FineWeb-Edu and render a 20B-token GPT-2 BPE dataset"
	)
	parser.add_argument("--dataset_repo", type=str, default=DEFAULT_DATASET_REPO)
	parser.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
	parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
	parser.add_argument("--val_tokens", type=int, default=DEFAULT_VAL_TOKENS)
	parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
	parser.add_argument("--seed", type=int, default=2357)
	parser.add_argument("--shuffle_buffer_size", type=int, default=DEFAULT_SHUFFLE_BUFFER_SIZE)
	parser.add_argument("--cache_dir", type=str, default=None)
	parser.add_argument("--output_dir", type=str, default=None)
	parser.add_argument("--resume", action="store_true", help="Append to exsiting bins by deterministic replay.")
	parser.add_argument("--overwrite", action="store_true", help="Delete existing train.bin/val.bin before writing.")
	parser.add_argument(
		"--data_files",
		nargs="+",
		default=None,
		help="Optional local parquet paths, globs, or URLs. If set, skip dataset repo loading."
	)
	return parser.parse_args()


def stable_ratio(key: str) -> float:
	digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
	value = int.from_bytes(digest, byteorder="big", signed=False)
	return value / float(1 << 64)


def pick_split(example, val_ratio: float) -> str:
	key = example.get("id") or example.get("url") or example["text"][:128]
	return "val" if stable_ratio(key) < val_ratio else "train"


def encode_example(example, enc):
	text = example.get("text", "")
	if not text:
		return None
	ids = enc.encode_ordinary(text)
	if not ids:
		return None
	ids.append(enc.eot_token)
	return np.asarray(ids, dtype=np.uint16)


def count_tokens(path: Path) -> int:
	if not path.exists():
		return 0
	size = path.stat().st_size
	if size % UINT16_BYTES != 0:
		raise  ValueError(f"{path} has a partial uint16 token at the end; remove it or rerun with --overwrite")
	return size / UINT16_BYTES


def remove_if_exists(path: Path):
	if path.exists():
		path.unlink()


def build_dataset(args):
	if args.data_files:
		print(
			"loading parquet data files in streaming mode; "
			f"target train={args.train_tokens:,} val={args.val_tokens:,} tokens"
		)
		dataset = load_dataset(
			"parquet",
			data_files={"train": args.data_files},
			split="train",
			streaming=True,
			cache_dir=args.cache_dir,
		)
	else:
		print(
			f"loading {args.dataset_repo} ({args.dataset_config}) in streaming mode; "
			f"target train={args.train_tokens:,} val={args.val_tokens:,} tokens"
		)
		dataset = load_dataset(
			args.dataset_repo,
			name=args.dataset_config,
			split="train",
			steaming=True,
			cache_dir=args.cache_dir
		)
	if args.shuffle_buffer_size > 0:
		dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
	return dataset


def main():
	args = parse_args()
	if not 0.0 < args.val_ratio < 1.0:
		raise ValueError("--val_ratio must be between 0 and 1")
	if args.resume and args.overflow:
		raise ValueError("--resume and --overwrite cannot be used together")
	
	output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).resolve().parent
	output_dir.mkdir(parents=True, exist_ok=True)
	train_path = output_dir / "train.bin"
	val_path = output_dir / "val.bin"
	meta_path = output_dir / "meta.pkl"

	if args.overwrite:
		remove_if_exists(train_path)
		remove_if_exists(val_path)
		remove_if_exists(meta_path)
	elif not args.resume and (train_path.exists() or val_path.exists()):
		raise FileExistsError("train.bin/val.bin already exist; use --resume or --overwrite")
	
	enc = tiktoken.get_encoding("gpt2")
	dataset = build_dataset(args)

	targets = {"train": args.train_tokens, "val": args.val_tokens}
	counts = {"train": count_tokens(train_path), "val": count_tokens(val_path) if args.resume else {"train": 0, "val": 0}}
	skip_tokens = {
		split: counts[split] if 0 < counts[split] < targets[split] else 0
		for split in ("train", "val")
	}
	docs = {"train": 0, "val": 0}

	if counts["train"] >= targets["train"] and counts["val"] >= targets["val"]:
		print("existing token bins already satisfy requested targets")
	else:
		mode = "ab" if args.resume else "wb"
		with open(train_path, mode) as train_f, open(val_path, mode) as val_f:
			files = {"train": train_f, "val": val_f}
			progress = tqdm(
				total=args.train_tokens + args.val_tokens,
				initial=min(counts["train"], targets["train"]) + min(counts["val"], targets["val"]),
				unit="tok",
				desc="writing fineweb-edu 20b"
			)

			for example in dataset:
				if counts["train"] >= targets["train"] and counts["val"] >= targets["val"]:
					break

				split = pick_split(example, args.val_ratio)
				if counts[split] >= targets[split]:
					continue

				ids = encode_example(example, enc)
				if ids is None:
					continue

				token_count = int(ids.size)
				if skip_tokens[split] > 0:
					if token_count > skip_tokens[split]:
						raise RuntimeError(
							f"cannot resume {split}: existing bin ends inside a document; "
							"rerun with --overwrite"
						)
					skip_tokens[split] -= token_count
					continue

				ids.tofile(files[split])
				counts[split] += token_count
				docs[split] += 1
				progress.update(token_count)
				progress.set_postfix(
					train_b=f"{counts['train'] / 1e9:.2f}",
					val_m=f"{counts["val"] / 1e6:.1f}"
				)

			progress.close()

	meta = {
		"vocab_size": PADDED_GPT2_VOCAB_SIZE,
		"tokenizer": "gpt2",
		"tokenizer_vocab_size": enc.n_vocab,
		"dataset_repo": args.dataset_repo,
		"dataset_config": args.dataset_config,
		"data_files": args.data_files,
		"train_tokens": counts["train"],
		"val_tokens": counts["val"],
		"train_docs_written_this_run": docs["train"],
		"val_docs_written_this_run": docs["val"],
		"val_ratio": args.val_ratio,
		"seed": args.seed,
		"shuffle_buffer_size": args.shuffle_buffer_size,
	}
	with open(meta_path, "wb") as f:
		pickle.dump(meta, f)
	
	print("done")
	print(f"train tokens: {counts['train']:,}")
	print(f"val tokens:   {counts['val']:,}")
	print(f"wrote: {train_path}")
	print(f"wrote: {val_path}")
	print(f"wrote: {meta_path}")


if __name__ == "__main__":
	main()
