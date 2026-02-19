"""
LoRA Fine-Tuning: MedGemma 1.5 on RSNA Abdominal Trauma Dataset
=================================================================
Fine-tunes google/medgemma-1.5-4b-it on the RSNA 2023 Abdominal Trauma
Detection dataset using Parameter-Efficient Fine-Tuning (LoRA via PEFT)
and TRL's SFTTrainer.

This transforms the generalist MedGemma into a trauma-specialist model
that has seen real labeled CT hemorrhage cases.

Designed to run on Google Colab (T4 16GB or A100 40GB).
Expected training time: ~2 hours on T4, ~45 min on A100.

Setup (run once in Colab):
    !pip install transformers>=4.47.0 peft>=0.10.0 trl>=0.8.0 bitsandbytes accelerate datasets

    from huggingface_hub import login
    login("hf_YOUR_TOKEN")

    from google.colab import drive
    drive.mount('/content/drive')

Usage:
    python scripts/lora_finetune.py \\
        --output_dir /content/drive/MyDrive/medgemma_lora \\
        --num_epochs 3 \\
        --max_samples 200
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "google/medgemma-1.5-4b-it"
DATASET_ID = "jherng/rsna-2023-abdominal-trauma-detection"

# LoRA targets for Gemma3 architecture (PaliGemma-style attention + MLP)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def build_training_examples(dataset_split, processor, max_samples: int = None):
    """
    Convert RSNA abdominal trauma dataset into MedGemma chat format.

    Actual dataset fields (jherng/rsna-2023-abdominal-trauma-detection):
        img_path    : str  — path to the CT slice image file
        extravasation: int — ClassLabel: 0=healthy, 1=injury (active bleeding)
        liver       : int  — ClassLabel: 0=healthy, 1=low grade, 2=high grade
        spleen      : int  — ClassLabel: 0=healthy, 1=low grade, 2=high grade
        kidney      : int  — ClassLabel: 0=healthy, 1=low grade, 2=high grade
        bowel       : int  — ClassLabel: 0=healthy, 1=injury
        any_injury  : bool — True if any organ has injury

    Label → response_json mapping:
        extravasation=1              → active hemorrhage description
        liver/spleen/kidney grade 2  → high-grade laceration (severe)
        liver/spleen/kidney grade 1  → low-grade laceration (moderate)
        any_injury=False             → no acute injury

    Returns:
        list of {"messages": [...]} dicts compatible with SFTTrainer
    """
    # Organ grade names for prompt generation
    GRADE_NAMES = {0: "healthy", 1: "low-grade injury", 2: "high-grade injury"}

    examples = []
    items = list(dataset_split)
    if max_samples:
        items = items[:max_samples]

    for item in items:
        try:
            img_path = item.get("img_path")
            if not img_path:
                continue

            # Load CT image from disk
            ct_image = Image.open(img_path).convert("RGB")

            # Read classification labels
            extravasation = int(item.get("extravasation", 0))  # 0=healthy,1=injury
            liver   = int(item.get("liver",   0))  # 0/1/2
            spleen  = int(item.get("spleen",  0))
            kidney  = int(item.get("kidney",  0))
            bowel   = int(item.get("bowel",   0))  # 0=healthy,1=injury
            any_injury = bool(item.get("any_injury", False))

            # Build organs_involved list (exclude healthy)
            organs_involved = []
            if liver   > 0: organs_involved.append(f"liver ({GRADE_NAMES[liver]})")
            if spleen  > 0: organs_involved.append(f"spleen ({GRADE_NAMES[spleen]})")
            if kidney  > 0: organs_involved.append(f"kidney ({GRADE_NAMES[kidney]})")
            if bowel   > 0: organs_involved.append("bowel (injury)")

            # Determine severity from worst organ grade + extravasation
            max_grade = max(liver, spleen, kidney)
            if max_grade == 2 or (extravasation == 1 and max_grade >= 1):
                severity = "severe"
            elif extravasation == 1 or max_grade == 1 or bowel == 1:
                severity = "moderate"
            else:
                severity = "none"

            # Build injury pattern description
            if not any_injury:
                injury_pattern = "No acute intraabdominal injury identified"
                bleeding_description = "No active extravasation or hemoperitoneum detected"
                differential = [
                    "No acute injury",
                    "Clinically correlate with exam findings"
                ]
            else:
                organ_str = ", ".join(organs_involved) if organs_involved else "abdomen"
                if extravasation == 1:
                    injury_pattern = f"Active arterial extravasation with solid organ injury: {organ_str}"
                    bleeding_description = "Active contrast extravasation consistent with ongoing hemorrhage"
                else:
                    injury_pattern = f"Solid organ laceration without active extravasation: {organ_str}"
                    bleeding_description = "Parenchymal laceration with hematoma; no active extravasation"
                differential = [
                    "Solid organ laceration with contained hematoma",
                    "Active arterial extravasation requiring intervention",
                    "Subcapsular hematoma"
                ]
                if bowel == 1:
                    differential.append("Bowel perforation with mesenteric injury")

            response_json = json.dumps({
                "injury_pattern": injury_pattern,
                "organs_involved": [o.split(" (")[0] for o in organs_involved],
                "bleeding_description": bleeding_description,
                "severity_estimate": severity,
                "differential_diagnosis": differential,
            }, indent=2)

            # MedGemma chat format: image + text interleaved in user turn
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ct_image},
                        {"type": "text", "text": (
                            "You are a trauma radiologist. Analyze this abdominal CT angiogram "
                            "slice for hemorrhage and solid organ injury. Respond in JSON with "
                            "keys: injury_pattern, organs_involved, bleeding_description, "
                            "severity_estimate, differential_diagnosis."
                        )}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_json}]
                }
            ]

            examples.append({"messages": messages})

        except Exception as e:
            print(f"[Dataset] Skipping example: {e}")
            continue

    print(f"[Dataset] Built {len(examples)} training examples.")
    return examples


def collate_fn(processor):
    """Return a collate function that applies the MedGemma chat template."""
    def _collate(batch):
        texts = []
        images_list = []

        for example in batch:
            # Apply chat template to format messages
            text = processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            # Collect all images from the messages
            imgs = []
            for msg in example["messages"]:
                for part in msg.get("content", []):
                    if part.get("type") == "image":
                        imgs.append(part["image"])
            images_list.append(imgs if imgs else [Image.new("RGB", (224, 224))])

        # Flatten images: processor expects a flat list
        flat_images = [img for imgs in images_list for img in imgs]

        inputs = processor(
            text=texts,
            images=flat_images if flat_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # For causal LM training, labels = input_ids (with padding masked)
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    return _collate


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    print(f"\n{'='*60}")
    print("MedGemma 1.5 LoRA Fine-Tuning on RSNA Trauma Dataset")
    print(f"{'='*60}\n")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    print(f"Device: {'CUDA — ' + torch.cuda.get_device_name(0) if cuda else 'CPU (slow)'}")

    # --- Load dataset ---
    print(f"\n[1/4] Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train", token=hf_token, trust_remote_code=True)
    print(f"      Total examples available: {len(dataset)}")

    # --- Load model with 4-bit quantization ---
    print(f"\n[2/4] Loading model: {MODEL_ID}")

    # Free any leftover VRAM from dataset loading before bringing in the model
    import gc
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    bnb_config = None
    if cuda:
        # T4 is Turing architecture — no native bfloat16, use float16 instead
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if cuda else torch.float32,  # float16 for T4, not bfloat16
        device_map="auto" if cuda else "cpu",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,  # load layer-by-layer to avoid RAM spike
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)

    # Ensure pad token exists
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- Apply LoRA ---
    print(f"\n[3/4] Applying LoRA (r=16, alpha=32, targets: {LORA_TARGET_MODULES})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Prepare training data ---
    print(f"\n[4/4] Preparing training examples (max_samples={args.max_samples})")
    train_examples = build_training_examples(dataset, processor, max_samples=args.max_samples)

    if not train_examples:
        raise ValueError("No training examples could be built. Check dataset format.")

    # --- Training configuration ---
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        learning_rate=args.learning_rate,
        fp16=cuda,   # T4 uses float16, not bfloat16
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        max_seq_length=2048,
    )

    # --- Train ---
    print(f"\nStarting training:")
    print(f"  Epochs:           {args.num_epochs}")
    print(f"  Batch size:       1 (effective: {8})")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Training samples: {len(train_examples)}")
    print(f"  Output dir:       {output_dir}\n")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=collate_fn(processor),
    )

    trainer.train()

    # --- Save adapter weights ---
    adapter_path = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"\n[Done] LoRA adapter saved to: {adapter_path}")
    print("To load for inference:")
    print(f"  model = AutoModelForImageTextToText.from_pretrained('{MODEL_ID}')")
    print(f"  model = PeftModel.from_pretrained(model, '{adapter_path}')")

    # Save training summary
    summary = {
        "base_model": MODEL_ID,
        "dataset": DATASET_ID,
        "num_epochs": args.num_epochs,
        "training_samples": len(train_examples),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "adapter_path": str(adapter_path),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {output_dir / 'training_summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune MedGemma 1.5 on RSNA trauma data")
    parser.add_argument("--output_dir", default="models/medgemma_lora",
                        help="Directory to save adapter weights")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max training examples (full dataset=206)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
