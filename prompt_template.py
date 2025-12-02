from typing import List, Dict, Literal, Optional
from PIL import Image
import os
import re
import torch
import torch.nn.functional as F
import copy

Role = Literal["system", "user", "assistant"]
Message = Dict[str, object]

# -------------------------------------------------------------------
# Helper: map prompt_mode string to k-shot count  ("zero", "1shot", "3shot", etc.)
# -------------------------------------------------------------------
def resolve_shot_count(mode: Optional[str]) -> int:
    m = (mode or "zero").lower()
    if m.startswith("1"):
        return 1
    if m.startswith("3"):
        return 3
    return 0  # default: zero-shot


# -------------------------------------------------------------------
# Generic few-shot injection (text-only)
# -------------------------------------------------------------------
def add_fewshot_examples(conversation: List[Message],
                         examples: List[Dict[str, str]],
                         k: int):
    """
    examples: list of {"user": "...", "assistant": "..."} items
    conversation: the final chat history (will be extended)
    k: number of examples to include
    """
    for ex in examples[:k]:
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": ex["user"]}],
        })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ex["assistant"]}],
        })


# You can fill these with CLEVR-style examples if you like
BINARY_FEWSHOT = [
    {
        "user": (
            "Given an image: Question: Are there any small red metal cubes?"
        ),
        "assistant": (
            "yes because there is at least one small red metal cube in the scene"
        ),
    },
    {
        "user": (
            "Given an image: Question: Is the number of blue spheres greater than the "
            "number of yellow cylinders?"
        ),
        "assistant": (
            "no because there are fewer blue spheres than yellow cylinders"
        ),
    },
]

COUNTING_FEWSHOT = [
    {
        "user": (
            "Given an image: Question: How many large green cubes are there?"
        ),
        "assistant": (
            "3 because there are exactly three large green cubes in the scene"
        ),
    },
    {
        "user": (
            "Given an image: Question: What number of red objects are to the left "
            "of the small yellow sphere?"
        ),
        "assistant": (
            "2 because there are two red objects located to the left of that sphere"
        ),
    },
]

ATTRIBUTE_FEWSHOT = [
    {
        "user": (
            "Given an image: Question: What color is the large cube to the left of the sphere?"
        ),
        "assistant": (
            "red because the large cube to the left of the sphere is red"
        ),
    },
    {
        "user": (
            "Given an image: Question: The cylinder in front of the small ball has what material?"
        ),
        "assistant": (
            "metal because that cylinder in front of the small ball is metal"
        ),
    },
]


# -------------------------------------------------------------------
# Shared system prompt + task instructions
# -------------------------------------------------------------------
system_prompt = """
You are a visual reasoning assistant for synthetic 3D scenes (CLEVR/CLEVR-X style).
Each image contains objects with the following attributes:

- shape: cube, sphere, cylinder
- color: gray, red, blue, green, brown, purple, cyan, yellow
- size: large, small
- material: rubber, metal

Your task is to answer questions strictly based on the image.
Internally, you may analyze, plan, and reason — but you must never reveal your reasoning steps.

Your final answer must always follow this format:

<answer> because <explanation>

Rules:
- Keep <answer> short and valid for the task (yes/no, integer, or attribute).
- <explanation> must cite visual evidence from the scene.
- Do not produce chain-of-thought, step-by-step reasoning, plans, or hidden analysis.
"""

instructions_yes_no = (
    "Given an IMAGE and a QUESTION, answer using this format:\n"
    "<answer> because <explanation>\n"
    "where <answer> is exactly 'yes' or 'no' in lowercase.\n"
)

instructions_counting = (
    "Given an IMAGE and a QUESTION, answer using this format:\n"
    "<number> because <explanation>\n"
    "where <number> is an integer between 0 and 10 (inclusive).\n"
)

instructions_attributes = (
    "Given an IMAGE and a QUESTION, answer about one of these attributes:\n"
    "- shape: cube, sphere, cylinder\n"
    "- color: gray, red, blue, green, brown, purple, cyan, yellow\n"
    "- size: large, small\n"
    "- material: rubber, metal\n"
    "Respond in the format:\n"
    "<attribute> because <explanation>\n"
    "Use exactly one attribute word as <attribute>.\n"
)


# -------------------------------------------------------------------
# Binary (yes/no) prompts
# -------------------------------------------------------------------
def prompt_binary_expl(
    image: Image.Image,
    question: str,
    prompt_mode: int = 0
) -> List[Message]:
    """
    Build a CLEVR-X style conversation for yes/no questions:
    model should output:  "<yes/no> because <short explanation>"
    """

    k = prompt_mode
    conversation: List[Message] = []

    # System message: global behavior
    conversation.append({
        "role": "system",
        "content": [{
            "type": "text",
            "text": system_prompt,
        }],
    })

    # Optional few-shot examples
    add_fewshot_examples(conversation, BINARY_FEWSHOT, k)

    instructions = (
        instructions_yes_no +
        f"\nQuestion: {question}"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instructions},
        ],
    })

    return conversation


# -------------------------------------------------------------------
# Counting prompts
# -------------------------------------------------------------------
def prompt_counting_expl(
    image: Image.Image,
    question: str,
    prompt_mode: int = 0,
) -> List[Message]:
    """
    Counting questions: model should output:
    "<number> because <explanation>"
    where <number> ∈ [0, 10].
    """

    k = prompt_mode
    conversation: List[Message] = []

    conversation.append({
        "role": "system",
        "content": [{
            "type": "text",
            "text": system_prompt,
        }],
    })

    # Optional few-shot examples
    add_fewshot_examples(conversation, COUNTING_FEWSHOT, k)

    instructions = (
        instructions_counting +
        f"\nQuestion: {question}"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instructions},
        ],
    })

    return conversation


# -------------------------------------------------------------------
# Attribute prompts
# -------------------------------------------------------------------
def prompt_attribute_expl(
    image: Image.Image,
    question: str,
    prompt_mode: int = 0,
) -> List[Message]:
    """
    Attribute questions: model should output:
    "<attribute> because <explanation>"

    Attributes must be from:
    - shape: cube, sphere, cylinder
    - color: gray, red, blue, green, brown, purple, cyan, yellow
    - size: large, small
    - material: rubber, metal
    """

    k = prompt_mode
    conversation: List[Message] = []

    conversation.append({
        "role": "system",
        "content": [{
            "type": "text",
            "text": system_prompt,
        }],
    })

    # Optional few-shot examples
    add_fewshot_examples(conversation, ATTRIBUTE_FEWSHOT, k)

    instructions = (
        instructions_attributes +
        f"\nQuestion: {question}"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instructions},
        ],
    })

    return conversation
