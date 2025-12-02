
import os
import re
from typing import List, Dict, Optional
from PIL import Image
import torch
import torch.nn.functional as F
import copy
from tqdm.auto import tqdm

from utils import load_custom_clevr, ClevrXExample, classify_clevr_question
from prompt_template import (prompt_binary_expl, prompt_counting_expl, prompt_attribute_expl)


##### INFERENCE #####

def split_explanation_answer(text: str):

    if not text:
        return "", ""

    parts = text.split("->", 1)  # split only on the first "->"

    if len(parts) == 2:
        explanation = parts[0].strip()
        answer = parts[1].strip()
    else:
        # no arrow found; treat whole text as explanation
        explanation = text.strip()
        answer = ""

    return explanation, answer



# -----------------------------
# Core generation (returns text + token_entropy)
# -----------------------------
from PIL import Image
import torch

def generate_answer(
    model,
    processor,
    conversation,
    max_new_tokens: int = 40,
):
    """
    - model: Qwen3-VL model (e.g., Qwen3VLForConditionalGeneration)
    - processor: corresponding AutoProcessor
    - image_path: path to image file
    - conversation: list of messages (without image), will be injected with the image
    - returns: generated text (str)
    """
    # Load image

    # Build model inputs from chat template
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # input_len = inputs["input_ids"].shape[1]

    # Generate output (no scores, no entropy)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # Take only the generated continuation (strip input prompt tokens)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    text = output_text[0]
    return text




def run_clevrx_task(
    model,
    processor,
    dataset_root: str,
    csv_path: str,
    n_samples: Optional[int] = None,
    num_k: int = 0,
    with_image: bool = False
):
    """
    Run CLEVR-X reasoning using Qwen3-VL on any CSV file
    (train_labels.csv, train_non_labels.csv, or test_non_labels.csv)
    
    - No split system.
    - No entropy.
    - Works with your simplified generate_answer().
    """

    # 1) Load dataset
    dataset = load_custom_clevr(dataset_root, csv_path)
    if n_samples is not None:
        dataset = dataset[:n_samples]

    print(f"Running CLEVR-X on {len(dataset)} samples (prompt_mode='{num_k}')...\n")

    results = []
    for i, s in tqdm(enumerate(dataset),desc="Evaluating CLEVR-X"):
        question = s.question
        gt = s.answer  # may be None (unlabeled/test)
        image_path = s.image_path
        img = Image.open(image_path).convert("RGB")
        # 2) Classify CLEVR-X question type
        qtype = classify_clevr_question(question)

        # 3) Build appropriate prompt
        if qtype == "binary":
            prompt = prompt_binary_expl(img, question, num_k, with_image)
        elif qtype == "counting":
            prompt = prompt_counting_expl(img, question, num_k, with_image)
        else:
            prompt = prompt_attribute_expl(img, question, num_k, with_image)

        # 4) Run Qwen model (no entropy)
        raw_pred = generate_answer(
            model=model,
            processor=processor,
            conversation=prompt,
        )

        pred_full = raw_pred.strip()


        # # 5) Split into <answer> and <explanation>
        # parts = re.split(r"\bbecause\b", pred_full, maxsplit=1, flags=re.I)
        # ans = parts[0].strip()
        # expl = parts[1].strip() if len(parts) > 1 else "explanation missing"
    
        # # 6) Normalize answer depending on the question type
        # if qtype == "binary":
        #     ans = ans.lower()
        #     if ans.startswith("yes"):
        #         ans = "yes"
        #     elif ans.startswith("no"):
        #         ans = "no"

        # elif qtype == "counting":
        #     m = re.search(r"\d+", ans)
        #     ans = m.group(0) if m else ans

        # else:  # attribute
        #     ans = ans.lower().strip()

        # Rebuild the "<answer> because <explanation>"
        # pred_full = f"{ans} because {expl}"
        explanation, label = split_explanation_answer(pred_full)
        # 7) Compute correctness if ground truth exists
        if gt is not None and gt != "":
            hit = int(label ==gt)
        else:
            hit = None

        # 8) Store result
        results.append({
            "idx": i,
            "question": question,
            "label": label,
            "ground_truth": gt,
            "explanation": s.explanation,
            "pred_full": pred_full,
            "correct": hit,
            "image": s.image_path,
            "num_k": num_k,
            "qtype": qtype,
        })

        if i % 200 == 0:
            print(f"Processed {i}/{len(dataset)}")

    # 9) Print dataset accuracy if available
    valid_hits = [r["correct"] for r in results if r["correct"] is not None]
    if valid_hits:
        acc = sum(valid_hits) / len(valid_hits)
        print(f"\nCLEVR-X Accuracy: {acc:.3f}")
    else:
        print("\nCLEVR-X: No ground-truth answers available for evaluation.")

    return results