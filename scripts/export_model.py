#!/usr/bin/env python3
"""
Export sklearn news classifier models to JSON for use by the Go classifier service.

Reads:
  cells/news_model/data/tree_model.joblib  (CountVectorizer, DecisionTreeClassifier)
  cells/news_model/data/svm_model.joblib   (TfidfVectorizer, LinearSVC)

Writes:
  cmd/classifier/tree_model.json
  cmd/classifier/svm_model.json
"""

import json
import os
import sys

import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_DIR = os.path.join(PROJECT_ROOT, "cells", "news_model", "data")
OUT_DIR = SCRIPT_DIR


def export_tree_model(tree_path: str, out_path: str) -> None:
    vec, clf = joblib.load(tree_path)
    t = clf.tree_

    # value shape is (n_nodes, 1, n_classes) — squeeze the middle dimension
    value_list = t.value[:, 0, :].tolist()

    data = {
        "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()},
        "ngram_min": int(vec.ngram_range[0]),
        "ngram_max": int(vec.ngram_range[1]),
        "binary": bool(vec.binary),
        "tree": {
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "value": value_list,
            "classes": [str(c) for c in clf.classes_],
        },
    }

    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB, vocab={len(data['vocabulary'])}, nodes={len(value_list)})")


def export_svm_model(svm_path: str, out_path: str) -> None:
    vec, clf = joblib.load(svm_path)

    data = {
        "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()},
        "ngram_min": int(vec.ngram_range[0]),
        "ngram_max": int(vec.ngram_range[1]),
        "sublinear_tf": bool(vec.sublinear_tf),
        "idf": vec.idf_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "classes": [str(c) for c in clf.classes_],
    }

    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB, vocab={len(data['vocabulary'])}, classes={data['classes']})")


def main() -> None:
    tree_src = os.path.join(MODEL_DIR, "tree_model.joblib")
    svm_src = os.path.join(MODEL_DIR, "svm_model.joblib")

    if not os.path.exists(tree_src):
        print(f"ERROR: {tree_src} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(svm_src):
        print(f"ERROR: {svm_src} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Exporting from {MODEL_DIR}")
    export_tree_model(tree_src, os.path.join(OUT_DIR, "tree_model.json"))
    export_svm_model(svm_src, os.path.join(OUT_DIR, "svm_model.json"))
    print("Done.")


if __name__ == "__main__":
    main()
