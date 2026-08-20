import gc
import glob
import os
import shutil
import torch
from google.colab import drive

# 1. Ensure Google Drive is Mounted
drive.mount('/content/drive')

# 2. Clear GPU VRAM Memory Cache
gc.collect()
torch.cuda.empty_cache()

# 3. Load trained checkpoint directly onto GPU
from unsloth import FastLanguageModel

print("1. Loading trained checkpoint onto GPU...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/plutus_outputs/checkpoint-90",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda:0",
)

# 4. Export GGUF locally
print("2. Converting and exporting to GGUF format locally...")
model.save_pretrained_gguf(
    "/content/plutus_export",
    tokenizer,
    quantization_method="q4_k_m",
)

# 5. Copy GGUF file to Google Drive
print("3. Copying GGUF to Google Drive...")
gguf_files = glob.glob("/content/**/*.gguf", recursive=True) + glob.glob(
    "/content/plutus_export/*.gguf"
)

if gguf_files:
    src = gguf_files[0]
    dst = "/content/drive/MyDrive/plutus-q4_k_m.gguf"
    shutil.copy(src, dst)
    print(f"\n🎉 SUCCESS! Model saved to Google Drive at: {dst}")
    print(f"File size: {os.path.getsize(dst) / (1024*1024*1024):.2f} GB")
else:
    print("Files in /content/plutus_export:")
    os.system("ls -lh /content/plutus_export/")
