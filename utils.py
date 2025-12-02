from __future__ import annotations
import os
import csv
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

@dataclass
class ClevrXExample:
    image_path: str
    question: str
    answer: Optional[str] = None        # None for unlabeled / test splits
    explanation: Optional[List[str]] =None        # list of explanation sentences
    sample_id: str = ""
    qtype: str = None  # to be filled later: 'binary', 'counting', 'attribute'



def load_custom_clevr(
    root: str,
    csv_path: str,
    training: bool = True,
) -> List[ClevrXExample]:

    results: List[ClevrXExample] = []

    # Candidate image directories
    img_dirs = [
        os.path.join(root, "train"),
        os.path.join(root, "test"),
        os.path.join(root, "train_non_labels"),
    ]

    # Load CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    #training
    if training:
        for row in rows:

            # --- Basic fields ---
            sample_id = str(row.get("id", ""))
            question = row.get("question", "")
            answer = row.get("answer") if "answer" in row else None
            file_name = row.get("file") or row.get("image") or row.get("filename")
            qtype = classify_clevr_question(question)
            # --- Locate image ---
            image_path: Optional[str] = None
            if file_name:
                for d in img_dirs:
                    cand = os.path.join(d, file_name)
                    if os.path.exists(cand):
                        image_path = cand
                        break

            # --- Parse explanation into List[str] ---
            expl_raw = row.get("explanation", "")
            explanation_list: List[str] = []

            if expl_raw:
                try:
                    parsed = ast.literal_eval(expl_raw)  # parse string into Python
                    if isinstance(parsed, list):
                        explanation_list = [str(s) for s in parsed]
                    else:
                        explanation_list = [str(parsed)]
                except Exception:
                    # fallback: use raw string as one-item list
                    explanation_list = [expl_raw]
            else:
                explanation_list = []

            # --- Build example ---
            results.append(
                ClevrXExample(
                    image_path=image_path or "",
                    question=question,
                    answer=answer,
                    explanation=explanation_list,
                    sample_id=sample_id,
                    qtype =qtype
                )
            )
    else:
        for row in rows:

            # --- Basic fields ---
            sample_id = str(row.get("id", ""))
            question = row.get("question", "")
            file_name = row.get("file") or row.get("image") or row.get("filename")
            qtype = classify_clevr_question(question)
            # --- Locate image ---
            image_path: Optional[str] = None
            if file_name:
                for d in img_dirs:
                    cand = os.path.join(d, file_name)
                    if os.path.exists(cand):
                        image_path = cand
                        break

            # --- Build example ---
            results.append(
                ClevrXExample(
                    image_path=image_path or "",
                    question=question,
                    sample_id=sample_id,
                    qtype =qtype
                )
            )

    return results


def classify_clevr_question(question: str) -> str:
    """
    Classify a CLEVR-X question into:
        - 'binary'     (yes/no questions: existence or comparisons)
        - 'counting'   (questions whose answer is an integer 0–10)
        - 'attribute'  (questions asking for shape, color, size, or material)

    The rules are derived from the patterns actually found in train_labels.csv.
    """

    q = str(question).strip().lower()
    if not q:
        return "attribute"

    # ----------------------------------------------------
    # 1. COUNTING questions: contain "how many" or
    #    "what number of" anywhere in the question
    # ----------------------------------------------------
    if "how many" in q or "what number" in q:
        return "counting"

    # ----------------------------------------------------
    # 2. BINARY questions: start with Is / Are / Does / Do
    #    (yes/no existence or comparison)
    # ----------------------------------------------------
    first = q.split()[0]
    if first in ("is", "are", "does", "do"):
        return "binary"
    if first in ("what"):
        return "attribute"
    # ----------------------------------------------------
    # 3. ATTRIBUTE questions: ask about color/shape/size/material
    #    using "what"/"how" patterns anywhere in the question
    # ----------------------------------------------------

    # e.g. "what color is", "there is ...; what size is it?",
    #      "the cube is what color?", "has what color?"
    if re.search(r"\bwhat\b.*\b(color|colour|shape|size|material)\b", q):
        return "attribute"

    # e.g. "is what color/size/shape/material?"
    if re.search(r"\b(color|colour|shape|size|material)\b.*\bis what\b", q):
        return "attribute"

    # e.g. "has what color/shape/size/material?"
    if re.search(r"\bhas what\b.*\b(color|colour|shape|size|material)\b", q):
        return "attribute"

    # e.g. "how big is it?", "how small is it?"
    if re.search(r"\bhow (big|small)\b", q):
        return "attribute"

    # e.g. "what is its color?", "what is the size/material/shape?"
    if re.search(r"what is (its|the) (color|colour|shape|size|material)", q):
        return "attribute"

    # e.g. "what is the cube made of?", "what is the small cube made of?"
    if "what is" in q and "made of" in q:
        return "attribute"

    # ----------------------------------------------------
    # 4. FALLBACK:
    #    any remaining question is treated as binary
    #    (this matches all remaining yes/no patterns in train_labels.csv)
    # ----------------------------------------------------
    return "binary"
