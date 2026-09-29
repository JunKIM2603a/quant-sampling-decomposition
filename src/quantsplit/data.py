"""Deterministic disjoint train splits; test text used only for duplicate audit."""
import hashlib
import json
import re
import unicodedata

SPLIT_VERSION = "nfkc-casefold-word5-jaccard080-v1"


def normalize_question(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def question_hash(text):
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def _shingles(text):
    words = re.findall(r"\w+|[^\w\s]", normalize_question(text))
    n = min(5, len(words))
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)} if n else set()


def make_splits(train, test, n=128, near_threshold=0.8):
    if n < 1 or not 0 < near_threshold <= 1:
        raise ValueError("invalid split parameters")
    normalized_test = {normalize_question(row["question"]): i for i, row in enumerate(test)}
    test_shingles = [_shingles(row["question"]) for row in test]
    index = {}
    for i, words in enumerate(test_shingles):
        for word in words:
            index.setdefault(word, set()).add(i)
    candidates = sorted(range(len(train)), key=lambda i: (question_hash(train[i]["question"]), i))
    selected, excluded, seen = [], [], set()
    for row_index in candidates:
        question = train[row_index]["question"]
        normal = normalize_question(question)
        reason, match = None, None
        if not normal:
            reason = "empty_question"
        elif normal in seen:
            reason = "duplicate_train"
        elif normal in normalized_test:
            reason, match = "exact_test_duplicate", normalized_test[normal]
        else:
            words = _shingles(question)
            possible = set().union(*(index.get(w, set()) for w in words)) if words else set()
            for test_index in sorted(possible):
                other = test_shingles[test_index]
                similarity = len(words & other) / len(words | other)
                if similarity >= near_threshold:
                    reason, match = "near_test_duplicate", test_index
                    break
        if reason:
            excluded.append({"train_row": row_index, "reason": reason, "test_row": match})
        else:
            seen.add(normal)
            selected.append({"row_index": row_index, "question_sha256": question_hash(question),
                             "question_id": f"gsm8k/train/{row_index}"})
        # Audit all candidates so exclusions are reproducible, without consulting answers.
    if len(selected) < 3*n:
        raise ValueError("insufficient unique training questions after duplicate audit")
    result = {"algorithm": SPLIT_VERSION, "near_threshold": near_threshold,
              "calibration": selected[:n], "development": selected[n:2*n],
              "pilot": selected[2*n:3*n], "excluded": excluded,
              "train_count": len(train), "test_count": len(test),
              "train_questions_sha256": hashlib.sha256("\n".join(question_hash(r["question"]) for r in train).encode()).hexdigest(),
              "test_questions_sha256": hashlib.sha256("\n".join(question_hash(r["question"]) for r in test).encode()).hexdigest()}
    return result


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()
