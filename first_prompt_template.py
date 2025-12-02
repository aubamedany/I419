from typing import List, Dict, Literal, Optional
from PIL import Image
import os
import re
import torch
import torch.nn.functional as F
import copy

Role = Literal["system", "user", "assistant"]
Message = Dict[str, object]


SYSTEM_PROMPT_BINARY = """
You are a visual reasoning assistant for synthetic 3D scenes. Think step-by-step but return only the final answer and a short explanation.
Each image contains objects with the 4 attributes (shape, color, size, material)
Given an IMAGE and a QUESTION, your task is to answer 'yes' or 'no' strictly based on the image.
Your final answer must always follow this format:
<explanation> -> <answer>
Rules:
- <answer> is exactly 'yes' or 'no' in lowercase.
- <explanation> is concise and directly supports the final answer.
"""

SYSTEM_PROMPT_ATTRIBUTE = """
You are a visual reasoning assistant for synthetic 3D scenes. Think step-by-step but return only the final answer and a short explanation.
Each image contains objects with the 4 attributes (shape, color, size, material)
Given an IMAGE and a QUESTION, your task is to answer about one of these attributes strictly based on the image.
    "- shape: cube, sphere, cylinder\n"
    "- color: gray, red, blue, green, brown, purple, cyan, yellow\n"
    "- size: large, small\n"
    "- material: rubber, metal\n"
Your final answer must always follow this format:
<explanation> -> <answer>
Rules:
- <answer> is exactly one attribute in lowercase.
- <explanation> is concise and directly supports the final answer.
"""

SYSTEM_PROMPT_COUNTING = """
You are a visual reasoning assistant for synthetic 3D scenes. Think step-by-step but return only the final answer and a short explanation.
Each image contains objects with the 4 attributes (shape, color, size, material)
Given an IMAGE and a QUESTION, your task is to answer a counting problem strictly based on the image.
Your final answer must always follow this format:
<explanation> -> <answer>
Rules:
- <answer> is exactly an integer between 0 and 10.
- <explanation> is concise and directly supports the final answer.
"""

BINARY_FEWSHOT = [
    {
        "file": "custom_dataset/custom_dataset/train/57698c6eb0068e6fc3aeba20b3a4981a.png",  # few-shot image file
        "question": "Are there the same number of big blue metal spheres that are in front of the big brown thing and red matte objects to the left of the large rubber block?",
        "explanation": "There are no big blue metal spheres that are in front of the big brown sphere and there are no red matte things which are on the left side of the large rubber block.",
        "answer": "yes",
    },
    {
        "file": "custom_dataset/custom_dataset/train/1a8979c6c2a8872b8ddeb7cfcc7178c6.png",  # few-shot image file
        "question": "There is a rubber sphere behind the cyan metal object; is its color the same as the large cube?",
        "explanation": "There is a green rubber sphere that is behind the cyan metal cylinder and there is a large green cube.",
        "answer": "yes",
    },
    {
        "file": "custom_dataset/custom_dataset/train/adb1df22bac87f8fe56adbae5490e99b.png",  # few-shot image file
        "question": "Is the number of red blocks in front of the tiny cyan metal object greater than the number of big yellow metallic balls that are behind the gray matte cylinder?",
        "explanation": "There are no red blocks which are in front of the tiny cyan metal sphere and there are no big yellow metallic balls which are behind the gray matte cylinder.",
        "answer": "no",
    },
    {
        "file": "custom_dataset/custom_dataset/train/461a572ba794af405bfe1f972e3bf0ec.png",  # few-shot image file
        "question": "Is the number of red blocks in front of the tiny cyan metal object greater than the number of big yellow metallic balls that are behind the gray matte cylinder?",
        "explanation": "There are no red blocks which are in front of the tiny cyan metal sphere and there are no big yellow metallic balls which are behind the gray matte cylinder.",
        "answer": "no",
    },
    {
        "file": "custom_dataset/custom_dataset/train/06bccea9ecad846ef319103ec32cd7cf.png",  # few-shot image file
        "question": "Are there the same number of small green shiny things that are in front of the green metallic thing and yellow cylinders to the left of the tiny brown rubber cylinder?",
        "explanation": "There are no small green shiny things which are in front of the green metallic cylinder and there is a yellow cylinder that is to the left of the tiny brown rubber cylinder.",
        "answer": "no",
    },
    {
        "file": "custom_dataset/custom_dataset/train/01739c509bc055e2da1fc5f4e87ed277.png",  # few-shot image file
        "question": "Are there any other things that are the same size as the green rubber sphere?",
        "explanation": "There are two small cubes, a small cylinder and a small sphere that have the same size as a green rubber sphere.",
        "answer": "yes",
    },
    {
        "file": "custom_dataset/custom_dataset/train/50de6007649aea401c9a5bcca8f60da8.png",  # few-shot image file
        "question": "Do the cyan metal object and the rubber cube that is right of the big rubber sphere have the same size?",
        "explanation": "There is a small cyan metal cube and there is a big rubber cube which is on the right side of the big rubber sphere.",
        "answer": "no",
    },
    {
        "file": "custom_dataset/custom_dataset/train/c3ea2c7a10ad3d80cd36d028ab8e29d8.png",  # few-shot image file
        "question": "Is there anything else of the same color as the tiny block?",
        "explanation": "There are a small yellow matte sphere and cylinder which have the identical color as a tiny block.",
        "answer": "yes",
    },
]

ATTRIBUTE_FEWSHOT = [
    {
        "file": "custom_dataset/custom_dataset/train/0a0e65fa046fe5162dbb262b30a22c8e.png",  # few-shot image file
        "question": "What is the shape of the big thing that is in front of the cyan metallic object and right of the tiny green shiny object?",
        "explanation": "There is a big ball in front of the cyan metallic block and to the right of the tiny green shiny cylinder.",
        "answer": "sphere",
    },
    {
        "file": "custom_dataset/custom_dataset/train/0a048855d4f88e73f3232383aeaeb897.png",  # few-shot image file
        "question": "What material is the cube that is in front of the tiny gray cube and behind the purple rubber block?",
        "explanation": "There is a rubber cube in front of the tiny gray cube and behind the purple rubber block.",
        "answer": "rubber",
    },
    {
        "file": "custom_dataset/custom_dataset/train/0a87898dfcd26c0ae4b7af4fbdd011bc.png",  # few-shot image file
        "question": "What material is the large cylinder left of the big cylinder on the right side of the yellow cylinder behind the large yellow cylinder?",
        "explanation": "There is a large metal cylinder that is to the left of the big cylinder that is right of the yellow cylinder that is behind the large yellow cylinder.'",
        "answer": "metal",
    },
    {
        "file": "custom_dataset/custom_dataset/train/607688ed44196e63390a713a348a832d.png",  # few-shot image file
        "question": "What size is the green thing that is behind the small shiny thing in front of the small blue object?",
        "explanation": "There is a tiny green cylinder which is behind the small shiny sphere that is in front of the small blue cylinder.",
        "answer": "small",
    },
    {
        "file": "custom_dataset/custom_dataset/train/6d515b67779707a53931980d2950c496.png",  # few-shot image file
        "question": "There is a large metallic thing that is the same shape as the large green rubber object; what is its color?",
        "explanation": "There is the large cyan metallic sphere that has the same shape as a large green rubber sphere.",
        "answer": "cyan",
    },
    {
        "file": "custom_dataset/custom_dataset/train/67b5147608fbc051f32d88222689e185.png",  # few-shot image file
        "question": "What is the color of the big metallic thing that is behind the large metal thing that is on the left side of the metallic cylinder that is to the right of the gray shiny object?",
        "explanation": "There is a big red metallic sphere that is behind the large metal cylinder that is left of the metallic cylinder that is right of the gray shiny cylinder.",
        "answer": "red",
    },
    {
        "file": "custom_dataset/custom_dataset/train/478591c5708ce418f6c90b74a7eff8f2.png",  # few-shot image file
        "question": "What is the shape of the small red object that is the same material as the tiny brown thing?",
        "explanation": "There is the small red metal cylinder that has the same material as a tiny brown sphere.",
        "answer": "cylinder",
    },
    {
        "file": "custom_dataset/custom_dataset/train/5f29460cbbf82c541e43e88ad2a1c2e9.png",  # few-shot image file
        "question": "There is a matte thing that is the same color as the large matte block; what is its size?",
        "explanation": "There is a big brown matte sphere that has the same color as a large matte block.",
        "answer": "large",
    },
]

COUNTING_FEWSHOT = [
    {
        "file": "custom_dataset/custom_dataset/train/12d593afbf4ae5d7168ad633336f09e3.png",  # few-shot image file
        "question": "What number of things are matte things that are in front of the ball or tiny cylinders that are in front of the large shiny ball?",
        "explanation": "There is a matte cylinder that is in front of the ball and there is a tiny cylinder which is in front of the large shiny ball.",
        "answer": "1",
    },
    {
        "file": "custom_dataset/custom_dataset/train/706fbab80d45831c18457e84141d217c.png",  # few-shot image file
        "question": "How many rubber spheres are to the right of the big metal object that is behind the large brown cylinder to the right of the metal ball?",
        "explanation": "There are two rubber spheres which are to the right of the big metal sphere that is behind the large brown cylinder that is right of the metal ball.",
        "answer": "2",
    },
    {
        "file": "custom_dataset/custom_dataset/train/3137a885651ca8d6728a3dcc7d49e628.png",  # few-shot image file
        "question": "What number of objects are big gray cubes or tiny objects in front of the tiny red rubber block?",
        "explanation": "There are two tiny spheres, three tiny cubes and two tiny cylinders which are in front of the tiny red rubber block.",
        "answer": "7",
    },
    {
        "file": "custom_dataset/custom_dataset/train/b62787f3a8029a2f33596355b3ce4b78.png",  # few-shot image file
        "question": "There is a gray metal cylinder; what number of red metallic things are right of it?",
        "explanation": "There are no red metallic things that are to the right of the gray metal cylinder.",
        "answer": "0",
    },
    {
        "file": "custom_dataset/custom_dataset/train/8eb1a1a6630c9bce828c06fe55a1ae3d.png",  # few-shot image file
        "question": "How many things are small metallic cubes or cylinders?",
        "explanation": "There are three cylinders.",
        "answer": "3",
    },
    {
        "file": "custom_dataset/custom_dataset/train/cd27db34d8bea1af360723898534ed7f.png",  # few-shot image file
        "question": "What number of other objects are there of the same size as the cyan rubber object?",
        "explanation": "There are two small cubes and two small cylinders which have the identical size as a cyan rubber cylinder.",
        "answer": "4",
    },
    {
        "file": "custom_dataset/custom_dataset/train/83975892ffe34efc1d35d35e34964980.png",  # few-shot image file
        "question": "How many green objects are matte spheres or big objects?",
        "explanation": "There are no green things.",
        "answer": "0",
    },
    {
        "file": "custom_dataset/custom_dataset/train/086dd3133e8a839831161dc84cdca030.png",  # few-shot image file
        "question": "How many other things are there of the same material as the purple ball?",
        "explanation": "There are four rubber cubes and a rubber cylinder that have the same material as a purple ball.",
        "answer": "5",
    },
]

FEWSHOT_INTRO = (
    "Please respond to the questions based on the given instructions "
    "and demonstrations below.\n"
)
INTRO = (
    "Please respond to the questions based on the given instructions and follow the format from demonstrations below.\n"
)
# -------------------------------------------------------------------
# Helper: map prompt_mode string to k-shot count  ("zero", "1shot", "3shot", etc.)
# -------------------------------------------------------------------
def _load_fewshot_image(file_path: str) -> Image.Image:
    """Load a few-shot image using FEWSHOT_IMAGE_ROOT + file."""
    return Image.open(file_path).convert("RGB")


def _format_expl_answer(explanation: str, answer: str) -> str:
    """Format '<explanation> -> <answer>'."""
    return f"{explanation.strip()} -> {answer.strip()}"


def _add_fewshot_with_images(
    conversation: List[Message],
    fewshot_list: List[Dict[str, str]],
    k: int,
):
    """
    Add up to k few-shot (image, question, explanation->answer) pairs
    to the conversation.
    """
    k = max(0, min(k, len(fewshot_list)))
    for ex in fewshot_list[:k]:
        fs_image = _load_fewshot_image(ex["file"])
        user_text = f"QUESTION: {ex['question']}"
        assistant_text = _format_expl_answer(ex["explanation"], ex["answer"])

        # user with image + question
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image", "image": fs_image},
                {"type": "text", "text": user_text},
            ],
        })

        # assistant with "<explanation> -> <answer>"
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        })

def _add_fewshot_text_only(
    conversation: List[Message],
    fewshot_list: List[Dict[str, str]],
    k: int,
):

    k = max(0, min(k, len(fewshot_list)))

    for ex in fewshot_list[:k]:
        user_text = f"QUESTION: {ex['question']}"
        assistant_text = _format_expl_answer(ex["explanation"], ex["answer"])

        # user with text-only question
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
            ],
        })

        # assistant with "<explanation> -> <answer>"
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        })


# -------------------------------------------------------------------
# Binary (yes/no) prompts
# -------------------------------------------------------------------

def prompt_counting_expl(
    image: Image.Image,
    question: str,
    num_shots: int = 0,
    with_image: bool = False,
) -> List[Message]:
    """
    Build a CLEVR-style conversation for counting questions.
    Model output format: '<explanation> -> <answer>' where <answer> is an integer string.
    """
    conversation: List[Message] = []

    conversation.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT_COUNTING}],
    })

    

    if num_shots > 0:
        conversation.append({
        "role": "user",
        "content": [{"type": "text", "text": INTRO}],
    })
        if with_image:
            _add_fewshot_with_images(conversation, COUNTING_FEWSHOT, num_shots)
        else:
            _add_fewshot_text_only(conversation, COUNTING_FEWSHOT, num_shots)

    user_text = (
        "Now answer this new QUESTION about the given IMAGE \n"
        f"QUESTION: {question}"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    })

    return conversation

# -------------------------------------------------------------------
# Counting prompts
# -------------------------------------------------------------------
def prompt_binary_expl(
    image: Image.Image,
    question: str,
    num_shots: int = 0,
    with_image: bool = False,
) -> List[Message]:
    """
    Build a CLEVR-style conversation for binary (yes/no) questions.
    Model output format: '<explanation> -> <answer>'.
    """
    conversation: List[Message] = []

    # system
    conversation.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT_BINARY}],
    })

    # intro before demonstrations
    

    # few-shot demonstrations
    if num_shots > 0:
        conversation.append({
        "role": "user",
        "content": [{"type": "text", "text": INTRO}],
    })
        if with_image:
            _add_fewshot_with_images(conversation, COUNTING_FEWSHOT, num_shots)
        else:
            _add_fewshot_text_only(conversation, COUNTING_FEWSHOT, num_shots)

    # actual query
    user_text = (
        "Now answer this new QUESTION about the given IMAGE \n"
        f"QUESTION: {question}"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    })

    return conversation

# -------------------------------------------------------------------
# Attribute prompts
# -------------------------------------------------------------------
def prompt_attribute_expl(
    image: Image.Image,
    question: str,
    num_shots: int = 0,
    with_image: bool = False,
) -> List[Message]:
    """
    Build a CLEVR-style conversation for attribute questions.
    Model output format: '<explanation> -> <answer>' where <answer> is a single attribute.
    """
    conversation: List[Message] = []

    conversation.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT_ATTRIBUTE}],
    })

    

    if num_shots > 0:
        conversation.append({
        "role": "user",
        "content": [{"type": "text", "text": INTRO}],
    })
        if with_image:
            _add_fewshot_with_images(conversation, COUNTING_FEWSHOT, num_shots)
        else:
            _add_fewshot_text_only(conversation, COUNTING_FEWSHOT, num_shots)

    user_text = (
        "Now answer this new QUESTION about the given IMAGE \n"
        f"QUESTION: {question}"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    })

    return conversation




