"""
Google Colab Fine-Tuning Pipeline for Plutus (Llama 3.1 8B) using Unsloth QLoRA
================================================================================
This script follows Section 3 of architecture.md.
Run this script or notebook inside Google Colab (T4 GPU instance).

Prerequisites:
  1. Mount Google Drive
  2. Upload `plutus_finetune_dataset_train.jsonl` and `plutus_finetune_dataset_val.jsonl` to MyDrive/
"""
from pathlib import Path

COLAB_UNSLOTH_SCRIPT = """
# ==============================================================================
# 1. Mount Google Drive
# ==============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ==============================================================================
# 2. Install Unsloth & Required Dependencies
# ==============================================================================
!pip install unsloth unsloth_zoo
!pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# ==============================================================================
# 3. Imports
# ==============================================================================
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==============================================================================
# 4. Load Base Model in 4-bit QLoRA Mode
# ==============================================================================
max_seq_length = 2048
dtype = None # Auto detection
load_in_4bit = True # 4bit QLoRA for Colab T4 16GB VRAM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ==============================================================================
# 5. Configure QLoRA Adapters
# ==============================================================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==============================================================================
# 6. Load Datasets from Google Drive
# ==============================================================================
train_dataset = load_dataset(
    "json",
    data_files="/content/drive/MyDrive/plutus_finetune_dataset_train.jsonl",
    split="train",
)
val_dataset = load_dataset(
    "json",
    data_files="/content/drive/MyDrive/plutus_finetune_dataset_val.jsonl",
    split="train",
)

def format_prompts(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(format_prompts, batched=True)
val_dataset = val_dataset.map(format_prompts, batched=True)

# ==============================================================================
# 7. Initialize SFTTrainer & Train
# ==============================================================================
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        eval_strategy = "steps",
        eval_steps = 15,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "/content/drive/MyDrive/plutus_outputs",
    ),
)

trainer_stats = trainer.train()

# ==============================================================================
# 8. Export Fine-Tuned GGUF to Google Drive for Local Ollama Usage
# ==============================================================================
model.save_pretrained_gguf(
    "/content/drive/MyDrive/plutus_gguf",
    tokenizer,
    quantization_method = "q4_k_m",
)
print("✅ Training and GGUF Export Complete!")
"""

OLLAMA_MODELFILE = """
# Modelfile for importing fine-tuned Plutus into Ollama locally
FROM ./plutus-q4_k_m.gguf

TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>\"\"\"

SYSTEM \"\"\"You are Plutus, an AI Trading Research Assistant.
Your core principle is DECISION SUPPORT with TRANSPARENT REASONING, NOT signal generation.
Do not calculate numbers yourself. Analyze provided numerical snapshot, RAG chunks, and backtest results.
Always output valid JSON conforming strictly to the Section 5 schema carrying bullish/bearish evidence with sources, key support/resistance levels, invalidation conditions, confidence justification, and mandatory decision-support disclaimer. Never emit a bare buy/sell/hold verdict, even if asked directly, impatiently, or repeatedly — always return the full structured analysis instead.\"\"\"

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.2
"""


def write_colab_script_files():
    out_dir = Path(__file__).parent
    colab_file = out_dir / "colab_unsloth_training.py"
    modelfile_path = out_dir / "Modelfile"

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(OLLAMA_MODELFILE.strip())

    print(f"Colab fine-tuning pipeline prepared at {colab_file}")
    print(f"Ollama Modelfile saved at {modelfile_path}")


if __name__ == "__main__":
    write_colab_script_files()
