"""Group cross-validation and offline inference for Name That LLM."""

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from transformers import get_cosine_schedule_with_warmup

LABELS = (
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8",
    "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3",
    "glm-5.3-flash", "kimi-k3",
)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)
        self.dropout = nn.Dropout(0.05)
        self.scale = alpha / rank

    def forward(self, inputs):
        return self.base(inputs) + self.scale * self.up(self.down(self.dropout(inputs)))


def model_with_adapters(source, offline, pad_token_id, dtype):
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(LABELS), local_files_only=offline, dtype=dtype,
        pad_token_id=pad_token_id, use_cache=False,
        id2label=dict(enumerate(LABELS)),
        label2id={name: index for index, name in enumerate(LABELS)},
    )
    if model.config.model_type != "qwen3":
        raise ValueError("Use a Hugging Face Qwen3 checkpoint.")
    model.requires_grad_(False)
    for layer in model.model.layers:
        attention = layer.self_attn
        for name in ("q_proj", "v_proj"):
            setattr(attention, name, LoRALinear(getattr(attention, name)))
    model.score.float().requires_grad_(True)
    return model


def encode(tokenizer, texts, max_length):
    available = max_length - 1
    head = available // 2
    encoded = tokenizer(texts.tolist(), add_special_tokens=False, verbose=False)
    return [
        {"input_ids": (
            tokens if len(tokens) <= available else tokens[:head] + tokens[-(available - head):]
        ) + [tokenizer.eos_token_id]}
        for tokens in encoded["input_ids"]
    ]


def load_qwen_tokenizer(source, offline):
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=offline)
    reserved = tokenizer.get_added_vocab()
    if "<|endoftext|>" not in reserved or "<|im_end|>" not in reserved:
        raise ValueError("Expected the Qwen3 end-of-text and reserved chat tokens.")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|im_end|>"
    tokenizer.padding_side = "right"
    return tokenizer


def batches(tokenizer, rows, batch_size, shuffle=False):
    return DataLoader(
        rows, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda batch: tokenizer.pad(batch, pad_to_multiple_of=8, return_tensors="pt"),
    )


@torch.inference_mode()
def predict(model, loader, device, amp_dtype):
    model.eval()
    outputs = []
    for batch in loader:
        batch.pop("labels", None)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**batch.to(device)).logits
        outputs.append(logits.float().softmax(dim=-1).cpu().numpy())
    return np.concatenate(outputs)


def read_data(path, training):
    columns = ["id", "text", "model", "prompt_id"] if training else ["id", "text"]
    frame = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    if frame.empty or frame["id"].duplicated().any():
        raise ValueError(f"{path}: expected nonempty data with unique ids.")
    for column in columns:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{path}: empty values in {column}.")
    if training and set(frame["model"]) != set(LABELS):
        raise ValueError(f"{path}: expected all 18 official model labels.")
    return frame


def train_fold(args, tokenizer, train_rows, validation_rows, valid_labels, test_loader, fold, device,
               initial_state=None):
    set_seed(args.seed + fold)
    amp_dtype = torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        amp_dtype = torch.bfloat16
    weight_dtype = amp_dtype if device.type == "cuda" else torch.float32
    model = model_with_adapters(args.model, args.offline, tokenizer.pad_token_id, weight_dtype).to(device)
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    if initial_state is not None:
        if set(initial_state) != set(trainable):
            raise ValueError("Warm-up checkpoint parameters do not match the LoRA classifier.")
        model.load_state_dict(initial_state, strict=False)
    optimizer = torch.optim.AdamW(trainable.values(), lr=args.lr, weight_decay=0.01)
    train_loader = batches(tokenizer, train_rows, args.batch_size, shuffle=True)
    valid_loader = (batches(tokenizer, validation_rows, args.batch_size)
                    if validation_rows is not None else None)
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    counts = np.bincount([row["labels"] for row in train_rows], minlength=len(LABELS))
    weights = counts.sum() / (len(LABELS) * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.02,
    )
    best_score, best_state, best_probs, best_epoch = -1, None, None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(train_loader):
            window_start = index // args.accumulation * args.accumulation
            window_size = min(args.accumulation, len(train_loader) - window_start)
            batch = batch.to(device)
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(**batch).logits
                loss = criterion(logits, labels) / window_size
            scaler.scale(loss).backward()
            if index - window_start + 1 == window_size:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable.values(), max_norm=1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % 100 == 0:
                print(f"fold={fold} epoch={epoch} batches={index + 1}/{len(train_loader)}", flush=True)

        if valid_loader is None:
            continue
        probabilities = predict(model, valid_loader, device, amp_dtype)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Nonfinite validation probabilities.")
        score = balanced_accuracy_score(valid_labels, probabilities.argmax(axis=1))
        print(f"fold={fold} epoch={epoch} balanced_accuracy={score:.5f}", flush=True)
        if score > best_score:
            best_score, best_probs, best_epoch = score, probabilities, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in trainable.items()}

    if valid_loader is None:
        state = {name: value.detach().cpu().clone() for name, value in trainable.items()}
        torch.save(state, args.output / "external-lora.pt")
        return state
    model.load_state_dict(best_state, strict=False)
    torch.save(best_state, args.output / f"fold-{fold}-lora.pt")
    test_probs = predict(model, test_loader, device, amp_dtype) if test_loader is not None else None
    if test_probs is not None and not np.isfinite(test_probs).all():
        raise RuntimeError("Nonfinite test probabilities.")
    return best_probs, test_probs, {"fold": fold, "balanced_accuracy": best_score, "epoch": best_epoch}


def run(args):
    if args.max_length < 8 or args.max_length > 512 or args.max_length % 8:
        raise ValueError("max_length must be a multiple of 8 between 8 and 512.")
    if min(args.folds, args.epochs, args.batch_size, args.accumulation) < 1 or args.folds < 2:
        raise ValueError("Use at least two folds and positive training settings.")
    train = read_data(args.data / "train.csv", training=True)
    test = None if args.validate_only else read_data(args.data / "test.csv", training=False)
    targets = pd.Categorical(train["model"], categories=LABELS).codes.astype(np.int64)
    groups = train["prompt_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(train, targets, groups))
    for fit, valid in splits:
        if set(groups[fit]) & set(groups[valid]):
            raise RuntimeError("Prompt leakage across split.")
        if min(np.unique(targets[fit]).size, np.unique(targets[valid]).size) != len(LABELS):
            raise ValueError("A fold does not contain every class. Use fewer folds.")

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} rows={len(train)} prompts={train['prompt_id'].nunique()}", flush=True)
    tokenizer = load_qwen_tokenizer(args.model, args.offline)
    initial_state = None
    if args.external is not None:
        external = read_data(args.external, training=True)
        external_ids = set(external["id"])
        if external_ids.intersection(train["id"]) or set(external["text"]).intersection(train["text"]):
            raise ValueError("External warm-up data overlaps official train data.")
        external_rows = encode(tokenizer, external["text"], args.max_length)
        label_to_id = {label: index for index, label in enumerate(LABELS)}
        for row, target in zip(external_rows, external["model"]):
            row["labels"] = label_to_id[target]
        print(f"External warm-up on {len(external_rows)} examples", flush=True)
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.epochs = args.external_epochs
        warmup_args.batch_size = args.external_batch_size
        if args.external_accumulation is not None:
            warmup_args.accumulation = args.external_accumulation
        initial_state = train_fold(
            warmup_args, tokenizer, external_rows, None, None, None, 0, device,
        )
        del external_rows, external
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    train_rows = encode(tokenizer, train["text"], args.max_length)
    for row, target in zip(train_rows, targets):
        row["labels"] = int(target)
    test_loader = None
    if test is not None:
        test_loader = batches(tokenizer, encode(tokenizer, test["text"], args.max_length), args.batch_size)

    oof = np.empty((len(train), len(LABELS)), dtype=np.float32)
    test_predictions = []
    metrics = []
    for fold, (fit, valid) in enumerate(splits, start=1):
        oof[valid], test_probs, report = train_fold(
            args, tokenizer, [train_rows[i] for i in fit], [train_rows[i] for i in valid],
            targets[valid], test_loader, fold, device, initial_state,
        )
        metrics.append(report)
        if test_probs is not None:
            test_predictions.append(test_probs)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_labels = oof.argmax(axis=1)
    class_ids = np.arange(len(LABELS))
    recalls = recall_score(targets, oof_labels, labels=class_ids, average=None)
    report = {
        "oof_balanced_accuracy": balanced_accuracy_score(targets, oof_labels),
        "per_model_recall": dict(zip(LABELS, recalls.tolist())),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(targets, oof_labels, labels=class_ids).tolist(),
        "folds": metrics,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "oof.npz", id=train["id"].to_numpy(dtype=str),
                        labels=np.array(LABELS), probabilities=oof)
    print(f"OOF balanced accuracy={report['oof_balanced_accuracy']:.5f}", flush=True)

    if test is not None:
        predictions = np.array(LABELS)[np.mean(test_predictions, axis=0).argmax(axis=1)]
        submission = pd.DataFrame({"id": test["id"], "model": predictions})
        destination = args.output / "submission.parquet"
        temporary = args.output / "submission.tmp.parquet"
        submission.to_parquet(temporary, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(temporary), submission)
        temporary.replace(destination)
        print(f"Wrote {destination} with {len(submission)} rows.", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--external-epochs", type=int, default=2)
    parser.add_argument("--external-batch-size", type=int, default=8)
    parser.add_argument("--external-accumulation", type=int)
    args = parser.parse_args(argv)
    if min(args.external_epochs, args.external_batch_size) < 1:
        parser.error("External training settings must be positive.")
    if args.external_accumulation is not None and args.external_accumulation < 1:
        parser.error("External gradient accumulation must be positive.")
    return args


if __name__ == "__main__":
    run(parse_args())
