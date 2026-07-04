from transformers import AutoConfig
import sys

models_to_test = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-7B-Instruct",
]

for model_name in models_to_test:
    try:
        c = AutoConfig.from_pretrained(model_name)
        print(f"OK: {model_name} -> model_type={c.model_type}")
    except Exception as e:
        print(f"FAIL: {model_name} -> {type(e).__name__}: {str(e)[:200]}")

print("transformers version:", __import__("transformers").__version__)
print("python version:", sys.version)
