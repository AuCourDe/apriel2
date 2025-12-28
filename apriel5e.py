def gradient_stability_probe(model, dataset, collator):
    if GRAD_PROBE_STEPS <= 0:
        return
    probe_samples = min(len(dataset), GRAD_PROBE_STEPS * BATCH_SIZE * max(1, GRAD_ACCUM))
    subset = dataset.select(range(probe_samples))
    print(f"🧪 Gradient probe: {GRAD_PROBE_STEPS} steps on {probe_samples} samples (LR={LR*0.1:.2e})")
    probe_args = TrainingArguments(
        output_dir=str(Path(OUTPUT_DIR) / "grad_probe"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=GRAD_PROBE_STEPS,
        learning_rate=LR * 0.1,
        warmup_ratio=0.0,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=1,
        save_steps=GRAD_PROBE_STEPS + 1,
        report_to="none",
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,
    )
    probe_trainer = Trainer(
        model=model,
        args=probe_args,
        train_dataset=subset,
        data_collator=collator,
    )
    probe_trainer.train()
    del probe_trainer

#!/usr/bin/env python3
"""
APRIEL-1.6-15B THINKER — POLISH QLoRA TRAINING (ETAP 2, AGGRESSIVE LORA, 12GB VRAM)
=======================================================================

✔ QLoRA 4-bit (nf4)
✔ Tylko lm_head + ostatnie warstwy dekodera (bez naruszania „myślenia”)
✔ Bez reasoning leakage
✔ Dynamic padding
✔ Gradient checkpointing
✔ Single-GPU friendly (12 GB VRAM)

INSTRUKCJA URUCHOMIENIA
-----------------------
1. Aktywuj środowisko: ``source venv/bin/activate``.
2. Ustaw docelowy katalog adaptera (np. ``OUTPUT_DIR=App/runs/apriel5e``), aby każda epoka
   zapisywała się do tego samego LoRA.
3. Wybierz epokę presetem ``EPOCH_NAME`` (``epoch1``…``epoch5``). Skrypt automatycznie
   dobierze dataset, LR i parametry LoRA; w razie potrzeby możesz nadpisać je zmiennymi.
4. Jeśli nie chcesz pytań interaktywnych, ustaw ``AUTO_RUN=1``. Sanity check można
   pomijać chwilowo przez ``SKIP_SANITY_CHECK=1``.

Przykład jednej epoki (np. epoka 3 – grammar):
```
source venv/bin/activate
OUTPUT_DIR=App/runs/apriel5e \
EPOCH_NAME=epoch3 \
AUTO_RUN=1 \
python apriel5e.py
```

Przykład pełnego przebiegu wszystkich epok na jednym adapterze:
```
source venv/bin/activate
for EP in epoch1 epoch2 epoch3 epoch4 epoch5; do
  OUTPUT_DIR=App/runs/apriel5e \
  EPOCH_NAME=$EP \
  AUTO_RUN=1 \
  python apriel5e.py
done
```

Upload LoRA na Hugging Face uruchamia się automatycznie po zakończeniu KAŻDEGO wywołania
skryptu (po zapisaniu adaptera). Aby upload nastąpił dopiero po ostatniej zaplanowanej
epoce, uruchom skrypt z ``HF_LORA_REPO`` ustawionym tylko podczas finalnego przebiegu
albo wykonaj wszystkie epoki sekwencyjnie bez ustawionego repozytorium i dopiero potem
odpal dodatkowe wywołanie w trybie uploadu.

Automatyczne, unikalne repozytorium HF:
- ustaw ``AUTO_NAME_HF_REPO=1`` oraz wskaż właściciela przez ``HF_LORA_OWNER=<twoj_login>``
  (lub przekaż ``HF_LORA_REPO=<twoj_login>/cokolwiek`` – zostanie wykorzystany prefiks
  przed ukośnikiem),
- repo zostanie nazwane ``apirelYYYYMMDD-HHMM-<epoka>``, więc każda epoka trafi do
  własnego prywatnego repo bez kolizji nazw.
"""

import os
import sys
import json
import torch
import importlib
import subprocess
import shutil
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # older versions
    from huggingface_hub import HfHubHTTPError  # type: ignore
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# ============================================================================
# BASIC CONFIG
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_URL = "https://huggingface.co/ServiceNow-AI/Apriel-1.6-15b-Thinker"
DEFAULT_LOCAL_DATASET = SCRIPT_DIR / "train_test_dataset/normalized/basic.json"


def _normalize_model_id(model_ref: str) -> str:
    """Accept full HF URL or repo id and return repo id."""
    ref = (model_ref or "").strip()
    ref = re.sub(r"^https://huggingface\.co/\s*", "", ref, flags=re.IGNORECASE)
    ref = ref.strip("/")
    return ref or "ServiceNow-AI/Apriel-1.6-15b-Thinker"


def _ensure_repo_id(repo_id: str) -> str:
    repo_id = (repo_id or "").strip()
    if not repo_id:
        raise ValueError("Identyfikator repozytorium Hugging Face nie może być pusty.")
    if "/" not in repo_id:
        raise ValueError("Identyfikator repozytorium musi mieć format 'uzytkownik/nazwa-repo'.")
    return repo_id


def _resolve_repo_owner(explicit_owner: str, repo_id: str) -> str:
    if explicit_owner:
        return explicit_owner.strip()
    if repo_id and "/" in repo_id:
        return repo_id.split("/", 1)[0].strip()
    return ""


def discover_last_layer_modules(model, num_layers: int = 2) -> List[str]:
    """Return lm_head plus q/v projections for the last N decoder layers."""
    target = ["lm_head"]
    q_proj = []
    v_proj = []
    for name, _ in model.named_modules():
        if name.endswith("self_attn.q_proj"):
            q_proj.append(name)
        elif name.endswith("self_attn.v_proj"):
            v_proj.append(name)
    q_proj.sort()
    v_proj.sort()
    if not q_proj or not v_proj:
        return target
    slice_count = min(num_layers, len(q_proj), len(v_proj))
    for q_name, v_name in zip(q_proj[-slice_count:], v_proj[-slice_count:]):
        target.append(q_name)
        target.append(v_name)
    return target


def validate_dataset_rows(rows: List[Dict[str, Any]]):
    if len(rows) < 100:
        print(f"⚠️ Uwaga: bardzo mały dataset ({len(rows)} przykładów). Trening może być niestabilny.")
    lengths = [len(example["text"].split()) for example in rows]
    if lengths:
        max_tokens = max(lengths)
        if max_tokens > MAX_LENGTH * 1.5:
            print(
                f"⚠️ Niektóre przykłady ({max_tokens} tokenów) znacząco przekraczają MAX_LENGTH={MAX_LENGTH}. "
                "Mogą zostać hard-trimowane."
            )


def backup_adapter(output_dir: str, epoch_name: str):
    src = Path(output_dir)
    if not src.exists() or not any(src.iterdir()):
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_dir = src.parent / f"{src.name}_backup_{epoch_name}_{stamp}"
    shutil.copytree(src, backup_dir, dirs_exist_ok=True)
    print(f"💾 Wykonano backup adaptera do: {backup_dir}")


def resolve_lora_targets(model, phase: str, layer_count: int) -> List[str]:
    if phase == "basic":
        return ["lm_head"]
    targets = discover_last_layer_modules(model, layer_count)
    if len(targets) == 1:
        print("⚠️ Ostrzeżenie: nie udało się odnaleźć warstw q/v – pozostaję przy samym lm_head.")
    return targets


def adjust_training_for_memory():
    global BATCH_SIZE, GRAD_ACCUM
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    total_mem = torch.cuda.get_device_properties(0).total_memory
    if total_mem < 14e9:
        print("⚠️ Wykryto mniejszą ilość VRAM – zmniejszam batch i zwiększam gradient accumulation.")
        BATCH_SIZE = max(1, BATCH_SIZE // 2)
        GRAD_ACCUM = max(1, GRAD_ACCUM * 2)


PL_RESPONSE_TOKEN = "<|pl_response|>"

# Default LoRA hyperparams (can be overridden per-epoch or via env)
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0

# Predefined per-epoch presets (dataset + key hyperparams)
EPOCH_PRESETS = {
    "epoch1": {
        "phase": "basic",
        "dataset": "train_test_dataset/normalized/basic.json",
        "learning_rate": 2e-5,  # MLP jest stabilniejsze, można użyć wyższego LR
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "lora_dropout": DEFAULT_LORA_DROPOUT,
    },
    "epoch2": {
        "phase": "basic",
        "dataset": "train_test_dataset/normalized/basic.json",
        "learning_rate": 2e-5,
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "lora_dropout": DEFAULT_LORA_DROPOUT,
    },
    "epoch3": {
        "phase": "grammar",
        "dataset": "train_test_dataset/normalized/grammar.json",
        "learning_rate": 3e-5,
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "lora_dropout": DEFAULT_LORA_DROPOUT,
    },
    "epoch4": {
        "phase": "grammar",
        "dataset": "train_test_dataset/normalized/grammar.json",
        "learning_rate": 3e-5,
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "lora_dropout": DEFAULT_LORA_DROPOUT,
    },
    "epoch5": {
        "phase": "advanced",
        "dataset": "train_test_dataset/normalized/advanced.json",
        "learning_rate": 2e-5,
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "lora_dropout": DEFAULT_LORA_DROPOUT,
    },
}

MODEL_ID = _normalize_model_id(os.environ.get("BASE_MODEL_ID", DEFAULT_MODEL_URL))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./apriel-pl-lora-part2")
_env_dataset = os.environ.get("LOCAL_DATASET_PATH", "").strip()

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "").strip()
HF_DATASET_FILE = os.environ.get("HF_DATASET_FILE", "").strip()
AUTO_NAME_HF_REPO = os.environ.get("AUTO_NAME_HF_REPO", "").lower() in {"1", "true", "yes"}
HF_LORA_OWNER = os.environ.get("HF_LORA_OWNER", "").strip()
AQUIRE_EXISTING_ADAPTER = os.environ.get("LOAD_EXISTING_ADAPTER", "").lower() in {"1", "true", "yes"}
BACKUP_ADAPTER = os.environ.get("BACKUP_ADAPTER", "").lower() in {"1", "true", "yes"}
SANITY_STRICT = os.environ.get("SANITY_STRICT", "1").lower() in {"1", "true", "yes"}
ENABLE_DATA_PARALLEL = os.environ.get("ENABLE_DATA_PARALLEL", "").lower() in {"1", "true", "yes"}
AUTO_CONFIRM_EXISTING = os.environ.get("AUTO_CONFIRM_EXISTING", "1").lower() in {"1", "true", "yes"}
AUTO_CONFIRM_TIMEOUT = int(os.environ.get("AUTO_CONFIRM_TIMEOUT", "30"))
_env_repo = os.environ.get("HF_LORA_REPO", "").strip()
try:
    HF_LORA_REPO = _ensure_repo_id(_env_repo) if _env_repo else ""
except ValueError:
    HF_LORA_REPO = ""
HF_UPLOAD_INTERVAL = int(os.environ.get("HF_UPLOAD_INTERVAL", "50"))
SKIP_SANITY_CHECK = os.environ.get("SKIP_SANITY_CHECK", "").lower() in {"1", "true", "yes"}
AUTO_RUN = os.environ.get("AUTO_RUN", "").lower() in {"1", "true", "yes"}
EPOCH_NAME = os.environ.get("EPOCH_NAME", "").strip().lower()
ACTIVE_EPOCH_PRESET = EPOCH_PRESETS.get(EPOCH_NAME)

if AUTO_NAME_HF_REPO:
    repo_owner = _resolve_repo_owner(HF_LORA_OWNER, HF_LORA_REPO)
    if not repo_owner:
        raise ValueError(
            "AUTO_NAME_HF_REPO=1 wymaga ustawienia HF_LORA_OWNER lub HF_LORA_REPO w formacie owner/repo."
        )
    epoch_label = EPOCH_NAME if EPOCH_NAME else (_env_phase or "custom")
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
    generated_repo = f"{repo_owner}/apirel{stamp}-{epoch_label}"
    HF_LORA_REPO = _ensure_repo_id(generated_repo)
    print(f"🔐 AUTO_NAME_HF_REPO → używam repozytorium {HF_LORA_REPO}")

if HF_LORA_REPO and not HF_TOKEN:
    raise ValueError("HF_LORA_REPO jest ustawione, ale brakuje HF_TOKEN – przerwanie przed treningiem.")

PHASE_CONFIG = {
    "basic": {
        "max_length": 256,
        "batch_size": 2,
        "grad_accum": 8,
        "epochs": 1,
        "lr": 2e-5,  # Wyższy LR dla MLP (stabilniejsze niż lm_head)
        "max_grad_norm": 1.0,
        "target_modules": [
            "model.language_model.layers.47.mlp.gate_proj",
            "model.language_model.layers.47.mlp.up_proj",
            "model.language_model.layers.47.mlp.down_proj",
        ],
    },
    "grammar": {
        "max_length": 384,
        "batch_size": 2,
        "grad_accum": 8,
        "epochs": 1,
        "lr": 3e-5,
        "max_grad_norm": 1.0,
        "target_modules": [
            "model.language_model.layers.47.mlp.gate_proj",
            "model.language_model.layers.47.mlp.up_proj",
            "model.language_model.layers.47.mlp.down_proj",
        ],
    },
    "advanced": {
        "max_length": 512,
        "batch_size": 2,
        "grad_accum": 8,
        "epochs": 1,
        "lr": 2e-5,
        "max_grad_norm": 1.0,
        "target_modules": [
            "model.language_model.layers.47.mlp.gate_proj",
            "model.language_model.layers.47.mlp.up_proj",
            "model.language_model.layers.47.mlp.down_proj",
        ],
    },
}

_env_phase = os.environ.get("TRAIN_PHASE", "basic").strip().lower()
TRAIN_PHASE = ACTIVE_EPOCH_PRESET["phase"] if ACTIVE_EPOCH_PRESET else _env_phase
PHASE = PHASE_CONFIG.get(TRAIN_PHASE, PHASE_CONFIG["basic"])

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", str(PHASE["max_length"])))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", str(PHASE["batch_size"])))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", str(PHASE["grad_accum"])))
EPOCHS = int(os.environ.get("EPOCHS", str(PHASE["epochs"])))
LR = float(
    os.environ.get(
        "LR",
        str(ACTIVE_EPOCH_PRESET["learning_rate"] if ACTIVE_EPOCH_PRESET else PHASE["lr"]),
    )
)
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", str(PHASE.get("max_grad_norm", 1.0))))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))  # 3% warmup
LR_SCHEDULER = os.environ.get("LR_SCHEDULER", "cosine")  # Cosine scheduler (bezpieczny z MLP)
GRAD_PROBE_STEPS = int(os.environ.get("GRAD_PROBE_STEPS", "0"))

LORA_R = 4
LORA_ALPHA = float(
    os.environ.get(
        "LORA_ALPHA",
        str(ACTIVE_EPOCH_PRESET["lora_alpha"] if ACTIVE_EPOCH_PRESET else DEFAULT_LORA_ALPHA),
    )
)
LORA_DROPOUT = float(
    os.environ.get(
        "LORA_DROPOUT",
        str(ACTIVE_EPOCH_PRESET["lora_dropout"] if ACTIVE_EPOCH_PRESET else DEFAULT_LORA_DROPOUT),
    )
)

LOCAL_DATASET_PATH = _env_dataset or (
    ACTIVE_EPOCH_PRESET["dataset"] if ACTIVE_EPOCH_PRESET else str(DEFAULT_LOCAL_DATASET)
)

# BEZPIECZNE: MLP ostatniej warstwy (47) - unika problemu z tied embeddings
# lm_head ma tie_word_embeddings=True co powoduje niestabilność
SAFE_TARGET_MODULES = [
    "model.language_model.layers.47.mlp.gate_proj",
    "model.language_model.layers.47.mlp.up_proj",
    "model.language_model.layers.47.mlp.down_proj",
]

SEED = 42

# ============================================================================
# DEPENDENCY CHECK (NO RESTART)
# ============================================================================

def ensure_deps():
    pkgs = [
        ("torch", "torch"),
        ("transformers", "transformers>=4.40"),
        ("peft", "peft>=0.11"),
        ("bitsandbytes", "bitsandbytes>=0.43"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate>=0.29"),
        ("PIL", "pillow"),
    ]
    for name, pip in pkgs:
        try:
            importlib.import_module(name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip])

# ============================================================================
# DATASET / HUB HELPERS
# ============================================================================


def download_dataset_from_hub() -> str:
    """Download dataset file from Hugging Face Hub (or reuse cached copy)."""
    print(f"📥 Downloading dataset {HF_DATASET_REPO}/{HF_DATASET_FILE} ...")
    downloaded_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=HF_DATASET_FILE,
        token=HF_TOKEN or None,
    )
    target = Path(LOCAL_DATASET_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloaded_path, target)
    print(f"✅ Dataset saved to {target}")
    return str(target)


def upload_folder_to_hf(path: str, repo_id: str, token: str, commit_message: str):
    """Upload entire folder to Hugging Face Hub."""
    if not token:
        print("⚠️ Skipping upload – HF_TOKEN not provided.")
        return
    if not repo_id:
        print("⚠️ Skipping upload – HF_LORA_REPO not provided.")
        return

    api = HfApi(token=token)
    print(f"📤 Uploading {path} to {repo_id} ({commit_message}) ...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except HfHubHTTPError as exc:
        print(f"⚠️ Nie udało się utworzyć repozytorium: {exc}. Próba uploadu do istniejącego repo...")
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print("✅ Upload finished.")


class HfLoraUploadCallback(TrainerCallback):
    """Uploads checkpoints to HF Hub whenever Trainer saves."""

    def __init__(self, repo_id: str, token: str):
        self.repo_id = repo_id
        self.token = token

    def on_save(self, args, state, control, **kwargs):
        if not self.token or not self.repo_id:
            return
        checkpoint_folder = kwargs.get("checkpoint_folder")
        if checkpoint_folder:
            upload_folder_to_hf(
                path=checkpoint_folder,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Checkpoint step {state.global_step}",
            )


class TrainingProgressCallback(TrainerCallback):
    """Thin logger for loss/grad diagnostics."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            print(f"📝 Epoka {state.epoch:.2f} | Krok {state.global_step} | Loss: {logs['loss']:.4f}")
        if "grad_norm" in logs:
            print(f"   ↳ Norma gradientu: {logs['grad_norm']:.3f}")


def timed_input(prompt: str, timeout: int, default: str | None = None) -> str:
    """Wait for input up to timeout seconds; return default on timeout."""
    if timeout <= 0 or default is None:
        return input(prompt)

    def _handler(signum, frame):
        raise TimeoutError

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        return input(prompt)
    except TimeoutError:
        print(f"\n⏱️ Brak reakcji przez {timeout}s – wybieram domyślnie opcję '{default}'.")
        return default
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)

# ============================================================================
# DATASET
# ============================================================================

def load_dataset_jsonl(path: str) -> Dataset:
    """Loads either JSONL or JSON-array dataset files into a Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        raw_data = f.read()

    stripped = raw_data.lstrip()
    if not stripped:
        raise ValueError(f"Dataset {path} is empty.")

    try:
        if stripped[0] == "[":
            # Full JSON array
            examples = json.loads(raw_data)
        else:
            # JSONL format (one object per line)
            examples = []
            for line in raw_data.splitlines():
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse dataset {path}: {exc}") from exc

    rows = []
    for ex in examples:
        if "instruction" in ex and "response" in ex:
            instruction = ex["instruction"].strip()
            response = ex["response"].strip()
            if instruction.startswith(PL_RESPONSE_TOKEN):
                prefixed_instruction = instruction
            else:
                prefixed_instruction = f"{PL_RESPONSE_TOKEN}{instruction}"
            rows.append({
                "text": f"{prefixed_instruction}\n\n{response}"
            })

    if not rows:
        raise ValueError("Dataset empty or malformed – missing 'instruction'/'response' pairs.")

    validate_dataset_rows(rows)
    return Dataset.from_list(rows)

# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize(dataset: Dataset, tokenizer):
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            add_special_tokens=True,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(
        _tok,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


# ============================================================================
# SANITY CHECKS
# ============================================================================

EN_SANITY_PROMPT = "Explain quantum physics in simple terms."
PL_SANITY_PROMPT = "Co to jest tryb przypuszczający w języku polskim i w jakich sytuacjach się go używa?"


def _generate_text(model, processor, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = processor(
        text=prompt,
        images=None,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    tokenizer = processor.tokenizer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_sanity_checks(model, processor):
    print("🧪 Running sanity checks...")
    en_response = _generate_text(model, processor, EN_SANITY_PROMPT)
    pl_response = _generate_text(
        model,
        processor,
        f"{PL_RESPONSE_TOKEN}{PL_SANITY_PROMPT}",
    )

    en_ok = "quantum" in en_response.lower()
    pl_has_diacritics = bool(re.search(r"[ąćęłńóśżź]", pl_response.lower()))
    pl_ok = pl_has_diacritics or " że " in pl_response.lower() or " tryb " in pl_response.lower()

    print(f"EN sanity response:\n{en_response}\n")
    print(f"PL sanity response:\n{pl_response}\n")

    if not en_ok:
        raise RuntimeError("Sanity check failed: English reasoning response no longer contains expected content.")
    if not pl_ok:
        raise RuntimeError("Sanity check failed: Polish response does not look Polish.")

    print("✅ Sanity checks passed.")

# ============================================================================
# TRAINING
# ============================================================================

def prompt_runtime_inputs():
    """Ask user for model URL, HF token and HF repo at runtime."""
    global MODEL_ID, HF_TOKEN, HF_LORA_REPO

    if AUTO_RUN:
        print("AUTO_RUN=1 → pomijam pytania interaktywne.")
        return

    current_model = MODEL_ID or DEFAULT_MODEL_URL
    model_input = input(f"Podaj adres modelu (HF URL lub repo id) [{current_model}]: ").strip()
    if model_input:
        MODEL_ID = _normalize_model_id(model_input)
    else:
        MODEL_ID = _normalize_model_id(current_model)

    if HF_TOKEN:
        token_input = input("Podaj token Hugging Face (ENTER aby użyć ustawionego): ").strip()
        if token_input:
            HF_TOKEN = token_input
    else:
        HF_TOKEN = input("Podaj token Hugging Face (wymagany do uploadu): ").strip()
        if not HF_TOKEN:
            print("⚠️ Nie podano tokenu – upload na HF nie będzie możliwy.")

    current_repo = HF_LORA_REPO or ""
    while True:
        prompt = f"Podaj repozytorium Hugging Face dla adaptera [{current_repo or 'wymagane'}]: "
        repo_input = input(prompt).strip()
        candidate = repo_input or current_repo
        try:
            HF_LORA_REPO = _ensure_repo_id(candidate)
            break
        except ValueError as exc:
            print(f"❌ {exc}")
            if not current_repo:
                continue



def determine_run_mode() -> str:
    """Decide whether to train, upload existing adapter, or abort."""
    out_path = Path(OUTPUT_DIR)
    if not out_path.exists() or not any(out_path.iterdir()):
        return "train"
    print(f"⚠️ Wykryto istniejący folder adaptera: {out_path}")
    print("Możesz: [T]renować ponownie, [U]ploadować istniejące pliki, [A]nulować.")
    while True:
        if AUTO_CONFIRM_EXISTING:
            choice = timed_input(
                f"Wybierz działanie [T/U/A] (auto-T za {AUTO_CONFIRM_TIMEOUT}s): ",
                AUTO_CONFIRM_TIMEOUT,
                default="t",
            ).strip().lower()
        else:
            choice = input("Wybierz działanie [T/U/A]: ").strip().lower()
        if choice in {"t", "train"}:
            return "train"
        if choice in {"u", "upload"}:
            return "upload"
        if choice in {"a", "abort", "s"}:
            return "abort"
        print("⚠️ Nie rozpoznano wyboru. Spróbuj ponownie.")


def resolve_dataset_path() -> str:
    """Return dataset path, download from HF only if local file missing and repo info provided."""
    local_path = Path(LOCAL_DATASET_PATH)
    if local_path.exists():
        print(f"📚 Używam lokalnego zbioru danych: {local_path}")
        return str(local_path)
    if HF_DATASET_REPO and HF_DATASET_FILE:
        return download_dataset_from_hub()
    raise FileNotFoundError(
        f"Nie znaleziono datasetu {local_path}. Podaj HF_DATASET_REPO/FILE lub umieść plik lokalnie."
    )


def main():
    ensure_deps()

    torch.manual_seed(SEED)

    prompt_runtime_inputs()

    run_mode = determine_run_mode()
    if run_mode == "upload":
        upload_folder_to_hf(
            path=OUTPUT_DIR,
            repo_id=HF_LORA_REPO,
            token=HF_TOKEN,
            commit_message="Upload istniejącego adaptera",
        )
        print("✅ Zakończono upload istniejącego adaptera.")
        return
    if run_mode == "abort":
        print("❎ Przerwano na życzenie użytkownika.")
        return

    # -------------------------
    # Download & load dataset
    # -------------------------
    dataset_path = resolve_dataset_path()
    dataset = load_dataset_jsonl(dataset_path)

    # -------------------------
    # Processor / tokenizer
    # -------------------------
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if PL_RESPONSE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [PL_RESPONSE_TOKEN]})
        tokenizer_added = True
    else:
        tokenizer_added = False

    # -------------------------
    # Quantization
    # -------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # -------------------------
    # Model
    # -------------------------
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    if tokenizer_added:
        model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # -------------------------
    # LoRA (SAFE TARGETS)
    # -------------------------
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=SAFE_TARGET_MODULES,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # -------------------------
    # Tokenize
    # -------------------------
    tokenized = tokenize(dataset, tokenizer)

    # -------------------------
    # Training data prep
    # -------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=0.01,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
    )

    # -------------------------
    # Collator
    # -------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[HfLoraUploadCallback(HF_LORA_REPO, HF_TOKEN)],
    )

    # -------------------------
    # Optional gradient probe
    # -------------------------
    gradient_stability_probe(model, tokenized, data_collator)

    # -------------------------
    # Train!
    # -------------------------
    trainer.train()

    if not SKIP_SANITY_CHECK:
        run_sanity_checks(model, processor)

    # -------------------------
    # Save final adaptor
    # -------------------------
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    upload_folder_to_hf(
        path=OUTPUT_DIR,
        repo_id=HF_LORA_REPO,
        token=HF_TOKEN,
        commit_message="Końcowy model",
    )

    print("\n✅ TRAINING FINISHED SUCCESSFULLY")
    print(f"Adapter saved to: {OUTPUT_DIR}")

# ... (rest of the code remains the same)
# ENTRY
# ============================================================================

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

    main()
