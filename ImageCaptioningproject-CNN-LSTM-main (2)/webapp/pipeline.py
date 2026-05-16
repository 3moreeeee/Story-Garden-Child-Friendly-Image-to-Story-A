from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import json
import pickle
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PROJECT_DIR = PROJECT_ROOT / "ImageCaptioningproject-CNN-LSTM-main"
STORY_DIR = PROJECT_ROOT / "story_lora_adapter"
QUIZ_DIR = MAIN_PROJECT_DIR / "quiz"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "Images"
IMAGES_FOOD_DIR = PROJECT_ROOT / "Images_food"


def _existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing file. Tried: {', '.join(str(p) for p in candidates)}")


CAPTION_WEIGHTS_PATH = _existing_path(
    MAIN_PROJECT_DIR / "best_model.h5",
    PROJECT_ROOT / "best_model.h5",
)
VOCAB_PATH = _existing_path(
    MAIN_PROJECT_DIR / "vocab.pkl",
    PROJECT_ROOT / "vocab.pkl",
)
CAPTIONS_TXT_PATH = _existing_path(
    DATA_DIR / "captions.txt",
    MAIN_PROJECT_DIR / "captions.txt",
)
CLIP_LABELS_PATH = MAIN_PROJECT_DIR / "clip_labels.pkl"


@dataclass
class PipelineResult:
    caption: str
    moderation_status: str
    moderation_score: float
    moderated_caption: str
    child_caption: str
    story: str
    quiz_text: str
    quiz_items: list[dict[str, Any]]
    labels: list[tuple[str, float]]
    notes: list[str]


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


@lru_cache(maxsize=1)
def load_vocab() -> dict[str, Any]:
    with open(VOCAB_PATH, "rb") as handle:
        vocab_blob = pickle.load(handle)

    if isinstance(vocab_blob, dict) and "word_to_index" in vocab_blob:
        word_to_index = vocab_blob["word_to_index"]
        max_length = int(vocab_blob.get("max_length", 100))
    else:
        raise ValueError("Unexpected vocab.pkl structure; expected a dict with word_to_index and max_length.")

    index_to_word = {index: word for word, index in word_to_index.items()}
    return {
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "max_length": max_length,
        "vocab_size": len(word_to_index) + 1,
    }


@lru_cache(maxsize=1)
def load_captions_index() -> dict[str, list[str]]:
    captions: dict[str, list[str]] = {}
    with open(CAPTIONS_TXT_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            image_name = parts[0].strip()
            caption = clean_text(",".join(parts[1:]))
            captions.setdefault(image_name, []).append(caption)

    for image_name, items in captions.items():
        captions[image_name] = [f"startseq {item} endseq" for item in items if item]
    return captions


def _build_caption_model(vocab_size: int, max_length: int):
    from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Input, LSTM
    from tensorflow.keras.models import Model

    image_input = Input(shape=(2048,), name="image_input")
    image_branch = Dropout(0.4)(image_input)
    image_branch = Dense(256, activation="relu")(image_branch)

    text_input = Input(shape=(max_length,), name="text_input")
    text_branch = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text_branch = Dropout(0.4)(text_branch)
    text_branch = LSTM(256)(text_branch)

    merged = Add()([image_branch, text_branch])
    merged = Dense(256, activation="relu")(merged)
    outputs = Dense(vocab_size, activation="softmax")(merged)

    model = Model(inputs=[image_input, text_input], outputs=outputs)
    return model


@lru_cache(maxsize=1)
def load_caption_system() -> dict[str, Any]:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50

    vocab = load_vocab()
    model = _build_caption_model(vocab["vocab_size"], vocab["max_length"])
    model.load_weights(str(CAPTION_WEIGHTS_PATH))
    model.trainable = False

    encoder = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    encoder.trainable = False

    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    return {
        "model": model,
        "encoder": encoder,
        **vocab,
    }


def _extract_image_features(image: Any):
    import numpy as np
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    system = load_caption_system()
    encoder = system["encoder"]

    resized = image.convert("RGB").resize((224, 224))
    array = img_to_array(resized)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    features = encoder.predict(array, verbose=0)
    return features[0]


def _fallback_caption(image: Any) -> str:
    try:
        from PIL import ImageStat

        resized = image.convert("RGB").resize((48, 48))
        stats = ImageStat.Stat(resized)
        red, green, blue = stats.mean[:3]
        brightness = (red + green + blue) / 3
        if brightness >= 180:
            mood = "bright and sunny"
        elif brightness >= 120:
            mood = "warm and cheerful"
        else:
            mood = "calm and cozy"
        if red > green and red > blue:
            color_hint = "red and rosy"
        elif green > red and green > blue:
            color_hint = "green and fresh"
        elif blue > red and blue > green:
            color_hint = "blue and peaceful"
        else:
            color_hint = "colourful"
        return f"A {mood}, {color_hint} scene full of lovely details and story ideas."
    except Exception:
        return "A lovely, child-friendly scene full of colours, shapes, and gentle details."


def _greedy_caption(feature_vec):
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    system = load_caption_system()
    model = system["model"]
    word_to_index = system["word_to_index"]
    index_to_word = system["index_to_word"]
    max_length = system["max_length"]

    start_token = word_to_index.get("startseq")
    end_token = word_to_index.get("endseq")
    if start_token is None or end_token is None:
        raise ValueError("Vocabulary does not contain startseq/endseq tokens.")

    sequence = [start_token]
    for _ in range(max_length):
        padded = pad_sequences([sequence], maxlen=max_length)
        preds = model.predict([feature_vec.reshape(1, 2048), padded], verbose=0)[0]
        next_index = int(np.argmax(preds))
        sequence.append(next_index)
        if next_index == end_token:
            break

    words = [index_to_word.get(idx, "") for idx in sequence]
    words = [word for word in words if word and word not in {"startseq", "endseq"}]
    return " ".join(words).strip()


def caption_image(image: Any, image_name: str | None = None) -> str:
    if image_name:
        captions_index = load_captions_index()
        if image_name in captions_index and captions_index[image_name]:
            return clean_text(captions_index[image_name][0].replace("startseq ", "").replace(" endseq", ""))

    try:
        feature_vec = _extract_image_features(image)
        raw_caption = _greedy_caption(feature_vec)
        return clean_text(raw_caption)
    except Exception:
        return clean_text(_fallback_caption(image))


_UNSAFE_TERMS = {
    "blood",
    "kill",
    "weapon",
    "knife",
    "gun",
    "fight",
    "hate",
    "drugs",
    "smoke",
    "alcohol",
    "fear",
    "violence",
    "adult",
    "nude",
}


def moderate_text(text: str) -> tuple[str, float, list[str]]:
    normalized = clean_text(text)
    terms = [term for term in _UNSAFE_TERMS if term in normalized]
    if terms:
        score = min(0.95, 0.55 + 0.1 * len(terms))
        return "needs_review", round(score, 2), terms
    return "safe", 0.05, []


_CHILD_FRIENDLY_REWRITES = [
    (r"\bman\b", "person"),
    (r"\bwoman\b", "person"),
    (r"\bboy\b", "child"),
    (r"\bgirl\b", "child"),
    (r"\bmen\b", "people"),
    (r"\bwomen\b", "people"),
    (r"\bhuge\b", "big"),
    (r"\bgiant\b", "big"),
    (r"\bscary\b", "spooky"),
    (r"\bdangerous\b", "careful"),
    (r"\bbeautiful\b", "bright"),
    (r"\bangry\b", "cross"),
    (r"\bjumping\b", "hopping"),
    (r"\bleaping\b", "hopping"),
    (r"\bracing\b", "playing"),
    (r"\briding\b", "travelling"),
    (r"\brunning\b", "dashing"),
    (r"\bclimbing\b", "scaling"),
    (r"\bswim trunks\b", "swim shorts"),
    (r"\bbodyboard\b", "surf board"),
    (r"\bfight\b", "play"),
    (r"\bkill\b", "help"),
    (r"\brun\b", "run happily"),
    (r"\bwalk\b", "stroll"),
]


def adapt_to_children(text: str) -> str:
    adapted = text
    for pattern, replacement in _CHILD_FRIENDLY_REWRITES:
        adapted = re.sub(pattern, replacement, adapted, flags=re.IGNORECASE)
    adapted = re.sub(r"\s+", " ", adapted).strip()
    return adapted


_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "at",
    "with",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "up",
    "down",
    "into",
    "from",
    "by",
    "this",
    "that",
    "it",
    "its",
    "as",
    "while",
    "very",
    "small",
    "little",
}


def _caption_keywords(caption: str, limit: int = 4) -> list[str]:
    words = re.findall(r"[a-z']+", caption.lower())
    keywords: list[str] = []
    for word in words:
        if word in _STOPWORDS:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords


def _fallback_story(caption: str) -> str:
    clean_caption_text = caption.strip().rstrip(".")
    keywords = _caption_keywords(clean_caption_text)
    keyword_text = ", ".join(keywords) if keywords else "gentle details"
    return "\n".join(
        [
            f"{clean_caption_text} opens a tiny door to a bright adventure.",
            f"The story begins with {keyword_text} and a gentle smile.",
            "A friendly helper arrives with soft footsteps and kind words.",
            "Together they explore a happy place full of colour and calm.",
            "Every step feels safe, warm, and full of wonder.",
            f"The little hero looks closely at the {keyword_text} and learns patience.",
            "A friend laughs softly and makes the day feel sweeter.",
            "The sky shines like a painted rainbow above the scene.",
            "A tiny problem is solved with calm thinking and care.",
            "Everyone feels proud of the gentle choice they made.",
            f"They pause to enjoy the {keyword_text} and share a smile.",
            "A butterfly floats by like a golden note in the air.",
            "Hope grows brighter as the adventure stays kind and safe.",
            "The story keeps its soft rhythm and its joyful heart.",
            "A warm hug says that everything is okay.",
            "The friends remember that kindness can make moments magical.",
            "They wave to the sun as the day slowly becomes evening.",
            "The memory they carry home feels shiny, sweet, and brave.",
            "The world feels friendlier because they cared for each other.",
            "And they all go home smiling, safe, and happy.",
        ]
    )


def _clean_story_lines(raw_story: str, target_lines: int = 20) -> str:
    lines = [re.sub(r"^\d+[\.\)]\s*", "", line).strip() for line in raw_story.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) >= target_lines:
        return "\n".join(lines[:target_lines])

    lines.extend(["And they lived happily ever after, full of joy and wonder."] * (target_lines - len(lines)))
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _load_story_model():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    base_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(STORY_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(STORY_DIR))
    model.eval()
    return model, tokenizer


def generate_story(caption: str) -> str:
    story_model_bundle = _load_story_model()
    if story_model_bundle is None:
        return _fallback_story(caption)

    try:
        import torch
    except Exception:
        return _fallback_story(caption)

    model, tokenizer = story_model_bundle
    user_prompt = (
        "Write a children's story of EXACTLY 20 lines based on this caption.\n\n"
        f"Caption: {caption}\n\n"
        "Requirements:\n"
        "- EXACTLY 20 lines, each a complete meaningful sentence\n"
        "- Simple, warm English for children aged 4-8\n"
        "- Clear structure: beginning (lines 1-5), middle (6-15), ending (16-20)\n"
        "- Magical, imaginative, emotionally gentle tone\n"
        "- NO violence, horror, fear, or adult themes\n"
        "- Stay faithful to the caption\n"
        "- End with joy, comfort, wonder, or belonging\n\n"
        "Output the 20 lines directly, one per line. No numbering. No JSON. No extra text."
    )

    inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, max_length=256)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=True,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _clean_story_lines(raw_text)


def _fallback_quiz(story: str) -> str:
    lines = [line for line in story.splitlines() if line.strip()]
    answer_focus = lines[0] if lines else "the story"
    summary_focus = _caption_keywords(answer_focus, limit=3)
    summary_text = ", ".join(summary_focus) if summary_focus else "the story"
    return "\n".join(
        [
            f"Q1: Who is the main idea of the story?\nA: {summary_text.title()}\nB: A machine\nC: A storm\nD: A cloud\nAnswer: A",
            f"Q2: What kind of feeling does the story have?\nA: Happy and calm\nB: Angry and loud\nC: Cold and scary\nD: Sleepy and dark\nAnswer: A",
            f"Q3: What is happening most in the story?\nA: A gentle adventure\nB: A race in space\nC: A silly argument\nD: A noisy battle\nAnswer: A",
            f"Q4: What can children learn from the story?\nA: Kindness matters\nB: Nobody helps\nC: Shouting is best\nD: Nothing is shared\nAnswer: A",
            f"Q5: Which part matches the story best?\nA: {answer_focus[:60]}\nB: A dragon fight\nC: A broken toy\nD: A stormy night\nAnswer: A",
        ]
    )


def parse_quiz_text(text: str) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^Q\d+:", line):
            if current and current.get("question"):
                questions.append(current)
            current = {"question": re.sub(r"^Q\d+:\s*", "", line), "options": {}, "answer": ""}
        elif line.startswith(("A:", "B:", "C:", "D:")) and current is not None:
            key = line[0]
            current["options"][key] = line[2:].strip()
        elif line.startswith("Answer:") and current is not None:
            current["answer"] = line.replace("Answer:", "").strip()[:1]

    if current and current.get("question"):
        questions.append(current)

    return questions


@lru_cache(maxsize=1)
def _load_quiz_model():
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(QUIZ_DIR), local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(QUIZ_DIR), local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception:
        return None


def generate_quiz(story: str) -> tuple[str, list[dict[str, Any]]]:
    quiz_bundle = _load_quiz_model()
    if quiz_bundle is None:
        raw = _fallback_quiz(story)
        return raw, parse_quiz_text(raw)

    import torch

    model, tokenizer, device = quiz_bundle
    prompt = "Generate a 5-question multiple choice quiz for this children story: " + story
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=256,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    items = parse_quiz_text(raw)
    if not items:
        raw = _fallback_quiz(story)
        items = parse_quiz_text(raw)
    return raw, items


def _load_optional_labels() -> dict[str, list[tuple[str, float]]]:
    if not CLIP_LABELS_PATH.exists():
        return {}
    with open(CLIP_LABELS_PATH, "rb") as handle:
        labels = pickle.load(handle)
    return labels


def dataset_labels_for(image_name: str) -> list[tuple[str, float]]:
    labels = _load_optional_labels()
    return labels.get(image_name, [])


def load_sample_images(limit: int = 12) -> list[Path]:
    candidates: list[Path] = []
    for folder in [IMAGES_DIR, IMAGES_FOOD_DIR]:
        if not folder.exists():
            continue
        for image_path in sorted(folder.glob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                candidates.append(image_path)
    return candidates[:limit]


def run_pipeline(image: Any, image_name: str | None = None) -> PipelineResult:
    caption = caption_image(image, image_name=image_name)
    moderation_status, moderation_score, flagged_terms = moderate_text(caption)
    moderated_caption = caption if moderation_status == "safe" else adapt_to_children(caption)
    child_caption = adapt_to_children(moderated_caption)
    if child_caption == caption:
        child_caption = adapt_to_children(f"{caption}.")
    story = generate_story(child_caption)
    quiz_text, quiz_items = generate_quiz(story)

    notes = []
    if flagged_terms:
        notes.append(f"Moderation flagged: {', '.join(flagged_terms)}")
    if moderation_status == "safe":
        notes.append("Caption is already child-friendly.")
    else:
        notes.append("Caption was rewritten using a gentle vocabulary pass.")
    if child_caption != caption:
        notes.append("Child-friendly version differs from the original caption.")
    else:
        notes.append("Child-friendly rewrite had no obvious changes, so the original wording was kept.")
    if image_name:
        notes.append(f"Dataset caption lookup used for {image_name}.")

    return PipelineResult(
        caption=caption,
        moderation_status=moderation_status,
        moderation_score=moderation_score,
        moderated_caption=moderated_caption,
        child_caption=child_caption,
        story=story,
        quiz_text=quiz_text,
        quiz_items=quiz_items,
        labels=[],
        notes=notes,
    )
