import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
import random
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION: DUAL H200 SXM ---
MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct" 
OUTPUT_DIR = "./qwen3_80b_rust_dual_h200" 
SEED = 42
LEARNING_RATE = 1e-4 
MAX_SEQ_LENGTH = 8192 
EPOCHS = 3            

# DUAL H200 SXM OPTIMIZATIONS (2x 141GB = 282GB total!)
# Much more headroom than single B200 (192GB)
# Can afford larger batches and potentially higher rank

PER_DEVICE_BATCH = 2          # Start here
GRADIENT_ACCUMULATION = 8     # Effective = 32
LORA_R = 64                   # This is fine
LORA_ALPHA = 128              # This is fine
DROPOUT = 0.05

# Flash Attention detection
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
    logger.info("✓ Flash Attention 2 detected")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.warning("⚠️ Flash Attention 2 not available, using PyTorch SDPA")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ----------------------------------------------------
# 2. DATA LOADING (12 Datasets)
# ----------------------------------------------------

def load_dataset_safe(dataset_name: str, split: str = "train"):
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split=split) 
        logger.info(f"✓ Loaded {dataset_name}: {len(ds)} examples")
        return ds
    except Exception as e:
        logger.error(f"✗ Failed to load {dataset_name}: {str(e)}")
        return None

swe_plus = load_dataset_safe("TuringEnterprises/SWE-Bench-plus-plus", split="train")
swe_pro = load_dataset_safe("ScaleAI/SWE-bench_Pro", split="verified")
if swe_pro is not None:
    swe_pro = swe_pro.repeat(8)

datasets_list = [
    # 1. Reasoning & Logic
    load_dataset_safe("snowmead/Strandset-Rust-Think"),
    # 2. Clean Code (Foundational)
    load_dataset_safe("Fortytwo-Network/Strandset-Rust-v1"),
    # 3. SQL Tools
    load_dataset_safe("kaxap/pg-wikiSQL-sql-instructions-80k"),
    # 4. Chat/Instruction
    load_dataset_safe("Tesslate/Rust_sharegpt"),
    # 5. Real World Bugs
    load_dataset_safe("hongliu9903/stack_edu_rust"),
    # 6. Agentic Workflow (Medium)
    swe_plus,
    # 7. Massive Code Corpus
    load_dataset_safe("ammarnasr/the-stack-rust-clean"),
    # 8. Diverse Code
    load_dataset_safe("Tesslate/Rust_Dataset"),
    # 9. Agentic Workflow (Hard)
    swe_pro,
    # 10. Framework Specifics
    load_dataset_safe("r1v3r/askama_axum_burn"),
    load_dataset_safe("h4/solfunmeme-dioxus-reports"),
    load_dataset_safe("quangtohe/postGIS_text2sql"),
]

datasets_list = [d for d in datasets_list if d is not None]
raw_dataset = concatenate_datasets(datasets_list)
logger.info(f"Total combined dataset size: {len(raw_dataset)}")

# ----------------------------------------------------
# 3. DATA FORMATTING & OPTIMIZED SPLIT
# ----------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_qwen_chat(sample: Dict[str, Any]) -> Dict[str, Any]:
    try:
        user_content = None
        assistant_content = None
        
        # 1. Standard Instruction
        if 'instruction' in sample and 'response' in sample:
            user_content = str(sample["instruction"]).strip()
            assistant_content = str(sample["response"]).strip()
        # 2. Code Prompt
        elif 'prompt' in sample and 'code' in sample:
            user_content = f"Generate Rust code for: {sample['prompt']}"
            assistant_content = str(sample["code"]).strip()
        # 3. SWE-Bench (Bug Fix)
        elif 'problem_statement' in sample and 'model_patch' in sample:
            repo = sample.get('repo', 'Unknown')
            user_content = f"Fix issue in {repo}:\n{sample['problem_statement']}"
            assistant_content = f"{sample['model_patch']}"
        # 4. Text-to-SQL
        elif 'text' in sample and 'sql' in sample:
            user_content = f"Generate PostgreSQL query for: {sample['text']}"
            assistant_content = str(sample["sql"]).strip()
        # 5. Dioxus Reports
        elif 'source_code' in sample and 'ast_analysis' in sample:
            user_content = f"Analyze Dioxus code:\n{sample['source_code'][:2000]}"
            assistant_content = str(sample['ast_analysis']).strip()
        # 6. ShareGPT
        elif 'messages' in sample and isinstance(sample['messages'], list):
            if len(sample['messages']) >= 2:
                user_content = sample['messages'][-2]['content']
                assistant_content = sample['messages'][-1]['content']
        # 7. The Stack
        elif 'content' in sample and 'path' in sample:
            if sample['path'].endswith('.rs'):
                user_content = f"Explain code in {sample['path']}:"
                assistant_content = str(sample["content"]).strip()

        if not user_content or not assistant_content:
            return {"text": ""}
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False)
        return {"text": text}
    
    except Exception:
        return {"text": ""}

logger.info("Formatting dataset...")
# Using 32 workers for high-core rental instances
formatted_data = raw_dataset.map(
    format_qwen_chat, 
    remove_columns=raw_dataset.column_names, 
    num_proc=32, 
    desc="Formatting"
)
formatted_data = formatted_data.filter(lambda x: x["text"] != "")
if len(formatted_data) == 0:
    raise ValueError("All samples were filtered out! Check format_qwen_chat logic")

# --- COST SAVING SPLIT: Capped Validation ---
logger.info("Splitting dataset with Cost Caps...")
shuffled_data = formatted_data.shuffle(seed=SEED)

# Validation set
val_size = 7500
train_size = len(shuffled_data) - val_size

# Fallback for tiny datasets
if val_size > len(shuffled_data) * 0.2: 
    val_size = int(len(shuffled_data) * 0.2)
    train_size = len(shuffled_data) - val_size

train_dataset = shuffled_data.select(range(train_size))
eval_dataset = shuffled_data.select(range(train_size, len(shuffled_data)))
logger.info(f"Train samples: {len(train_dataset)} | Validation samples: {len(eval_dataset)}")


# ----------------------------------------------------
# 4. MODEL LOADING (NATIVE BF16)
# ----------------------------------------------------

logger.info("Loading Model in Native BFloat16 for Dual H200 SXM...")
logger.info(f"Total VRAM available: 2x 141GB = 282GB")

# NO quantization needed - plenty of VRAM!
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",  # Will automatically split across both GPUs
    torch_dtype=torch.bfloat16,  
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa"
)

# Mandatory for 80B size optimization
model.gradient_checkpointing_enable() 
# Disabled torch.compile for first run - enable later if stable
# model = torch.compile(model, mode="max-autotune")
model.config.use_cache = False

logger.info(f"Model loaded. Memory allocated across GPUs:")
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved = torch.cuda.memory_reserved(i) / 1e9
    logger.info(f"  GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


# ----------------------------------------------------
# 5. LORA CONFIG
# ----------------------------------------------------

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# ----------------------------------------------------
# 6. TRAINER
# ----------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=25,
    
    # Evaluation & Saving
    eval_strategy="steps",
    eval_steps=125,
    save_strategy="steps",
    save_steps=125,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    torch_compile=False,  # Disabled for stability
    
    # Multi-GPU settings
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    packing=True,
    dataset_text_field="text",
)

logger.info("="*70)
logger.info("STARTING DUAL H200 SXM TRAINING RUN")
logger.info(f"Configuration: {PER_DEVICE_BATCH} batch × 2 GPUs × {GRADIENT_ACCUMULATION} accum = {PER_DEVICE_BATCH * 2 * GRADIENT_ACCUMULATION} effective batch")
logger.info("="*70)

trainer.train()

logger.info(f"Saving Final Adapter to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
logger.info("DONE.")