import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def language_shape(text):
    def replace(match):
        word = match.group()
        if word.lower() in ENGLISH_STOP_WORDS:
            return word
        pattern = "X" if word.isupper() else "T" if word[0].isupper() else "x"
        return pattern * min(len(word), 12)
    return WORD.sub(replace, text)


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--external", default="external_matched.csv")
parser.add_argument("--external-only", action="store_true")
args = parser.parse_args()
train = pd.read_csv(args.data / "train.csv", dtype=str, keep_default_na=False)
external = pd.read_csv(args.data / args.external, dtype=str, keep_default_na=False)
test = pd.read_csv(args.data / "test.csv", dtype=str, keep_default_na=False)
labels = np.array(sorted(train["model"].unique()))
assert set(external["model"]) == set(labels)
assert not set(train["text"]) & set(external["text"])
y = np.searchsorted(labels, train["model"])
external_y = np.searchsorted(labels, external["model"])
splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
# StratifiedGroupKFold codes groups based on first appearance of labels,
# so re-use the Qwen label order to guarantee exactly the original split.
qwen_order = np.array([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8",
    "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
])
split_y = pd.Categorical(train["model"], categories=qwen_order).codes
variants = ("external",) if args.external_only else ("official", "external")
output = {tag: np.zeros((len(train), len(labels))) for tag in variants}
test_output = {tag: [] for tag in output}
for n, (fit, valid) in enumerate(splitter.split(train, split_y, train["prompt_id"]), start=1):
    for tag in output:
        fit_text = train["text"].iloc[fit].tolist()
        fit_labels = y[fit]
        weights = np.ones(len(fit))
        if tag == "external":
            fit_text += external["text"].tolist()
            fit_labels = np.concatenate([fit_labels, external_y])
            weights = np.concatenate([weights, np.full(len(external), 0.35)])
        rest = train["text"].iloc[valid].tolist() + test["text"].tolist()
        raw = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False, min_df=2,
                              max_features=125_000, sublinear_tf=True, dtype=np.float32)
        shape = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), lowercase=False, min_df=2,
                                max_features=75_000, sublinear_tf=True, dtype=np.float32)
        x = hstack([raw.fit_transform(fit_text), 1.2 * shape.fit_transform(map(language_shape, fit_text))], format="csr")
        z = hstack([raw.transform(rest), 1.2 * shape.transform(map(language_shape, rest))], format="csr")
        clf = LinearSVC(C=2, random_state=42)
        clf.fit(x, fit_labels, sample_weight=weights)
        probs = softmax(clf.decision_function(z) * 4, axis=1)
        output[tag][valid] = probs[:len(valid)]
        test_output[tag].append(probs[len(valid):])
        print(f"fold={n} {tag} score={balanced_accuracy_score(y[valid], probs[:len(valid)].argmax(1)):.5f}", flush=True)

args.output.mkdir(parents=True, exist_ok=True)
for tag, probs in output.items():
    order = np.searchsorted(labels, qwen_order)
    print(f"OOF {tag} score={balanced_accuracy_score(y, probs.argmax(1)):.5f}", flush=True)
    np.savez_compressed(args.output / f"{tag}.npz", id=train["id"].to_numpy(dtype=str),
                        labels=qwen_order, probabilities=probs[:, order],
                        test_id=test["id"].to_numpy(dtype=str),
                        test_probabilities=np.mean(test_output[tag], axis=0)[:, order])
