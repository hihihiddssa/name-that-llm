import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description="Select reproducible subsets from the public open pool.")
parser.add_argument("--official-train", type=Path, required=True)
parser.add_argument("--pool", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
official = pd.read_csv(args.official_train, dtype=str)
root = args.output
root.mkdir(parents=True, exist_ok=True)
cols = [
    "id", "text", "model", "source_dataset", "source_license", "prompt_canon",
    "gen_route", "gen_served_model", "n_words", "flag_kb_proxy",
    "flag_kb_proxy_after_ruling", "flag_gemini37_routing_risk", "flag_near_train_prompt",
    "flag_train_prompt_regen",
]
dataset = args.pool
pool = pd.read_parquet(dataset, columns=cols)
labels = set(official["model"])
eligible = pool.loc[
    pool["model"].isin(labels)
    & pool["source_license"].isin(["Attribution 4.0 International (CC BY 4.0)", "CC-BY-SA-4.0", "CC0-1.0"])
    & pool["prompt_canon"].fillna("").ne("")
    & pool["n_words"].between(35, 275)
    & ~pool["gen_route"].fillna("").str.contains(r"kaggle|^proxy$", case=False, regex=True)
    & pool[[key for key in cols if key.startswith("flag_")]].eq(False).all(axis=1)
].copy()

def normalized_digest(text):
    stable = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()

official_hashes = set(official["text"].map(normalized_digest))
eligible["hash"] = eligible["text"].map(normalized_digest)
eligible = eligible.loc[~eligible["hash"].isin(official_hashes)]
ambiguous = eligible.groupby("hash")["model"].transform("nunique").ne(1)
eligible = eligible.loc[~ambiguous].drop_duplicates("hash")
eligible["reported_served_id"] = eligible["gen_served_model"].fillna("").ne("")
eligible["distance_from_median"] = (eligible["n_words"] - official["text"].str.split().str.len().median()).abs()
eligible = eligible.sort_values(
    ["reported_served_id", "distance_from_median", "id"], ascending=[False, True, True],
).drop_duplicates(["prompt_canon", "model"])
full = eligible.groupby("prompt_canon")["model"].transform("nunique").eq(len(labels))
panel = eligible.loc[full].sort_values(["prompt_canon", "model"]).copy()
assert len(panel) > 0 and panel["id"].is_unique and panel["hash"].is_unique
assert panel.groupby("prompt_canon")["model"].nunique().eq(len(labels)).all()
panel = panel.rename(columns={"prompt_canon": "prompt_id"})
panel[["id", "text", "model", "prompt_id"]].to_csv(root / "external_matched.csv", index=False)
panel[["id", "model", "prompt_id", "source_dataset", "source_license", "gen_route", "gen_served_model"]].to_csv(
    root / "external_manifest.csv", index=False,
)
eligible["eligible_panel_labels"] = eligible.groupby("prompt_canon")["model"].transform("nunique")
wide = eligible.loc[eligible["eligible_panel_labels"].ge(12)].copy()
per_class = min(1500, int(wide.groupby("model").size().min()))
assert per_class > 0 and set(wide["model"]) == labels
wide = wide.sample(frac=1, random_state=42).sort_values(
    ["eligible_panel_labels"], ascending=False, kind="stable",
).groupby("model", sort=False).head(per_class)
wide = wide.sort_values(["prompt_canon", "model"]).rename(columns={"prompt_canon": "prompt_id"})
wide[["id", "text", "model", "prompt_id"]].to_csv(root / "external_wide.csv", index=False)
wide[["id", "model", "prompt_id", "source_dataset", "source_license", "gen_route", "gen_served_model"]].to_csv(
    root / "external_wide_manifest.csv", index=False,
)
bulk = eligible.sample(frac=1, random_state=51).sort_values(
    ["eligible_panel_labels", "reported_served_id"], ascending=False, kind="stable",
).groupby(["model", "source_dataset"], sort=False).head(850)
bulk_size = min(2800, int(bulk.groupby("model").size().min()))
assert bulk_size > 0 and set(bulk["model"]) == labels
bulk = bulk.groupby("model", sort=False).head(bulk_size)
bulk = bulk.sort_values(["prompt_canon", "model"]).rename(columns={"prompt_canon": "prompt_id"})
bulk[["id", "text", "model", "prompt_id"]].to_csv(root / "external_bulk.csv", index=False)
bulk[["id", "model", "prompt_id", "source_dataset", "source_license", "gen_route", "gen_served_model"]].to_csv(
    root / "external_bulk_manifest.csv", index=False,
)
expanded = eligible.sample(frac=1, random_state=66).sort_values(
    ["eligible_panel_labels", "reported_served_id"], ascending=False, kind="stable",
).groupby(["model", "source_dataset"], sort=False).head(1800)
expanded = expanded.groupby("model", sort=False).head(5000)
expanded = expanded.sort_values(["prompt_canon", "model"]).rename(columns={"prompt_canon": "prompt_id"})
assert set(expanded["model"]) == labels
expanded[["id", "text", "model", "prompt_id"]].to_csv(root / "external_expanded.csv", index=False)
expanded[["id", "model", "prompt_id", "source_dataset", "source_license", "gen_route", "gen_served_model"]].to_csv(
    root / "external_expanded_manifest.csv", index=False,
)
print(json.dumps({
    "pool_rows": len(pool), "usable_rows": len(panel), "groups": panel["prompt_id"].nunique(),
    "sources": panel["source_dataset"].value_counts().to_dict(),
    "wide_rows": len(wide), "wide_per_class": per_class, "wide_groups": wide["prompt_id"].nunique(),
    "bulk_rows": len(bulk), "bulk_per_class": bulk_size, "bulk_groups": bulk["prompt_id"].nunique(),
    "expanded_rows": len(expanded), "expanded_groups": expanded["prompt_id"].nunique(),
    "expanded_counts": expanded["model"].value_counts().to_dict(),
}, ensure_ascii=False, indent=2))
