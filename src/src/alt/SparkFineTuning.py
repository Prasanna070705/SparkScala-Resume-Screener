import os
import random
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import StringType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import time

# ====================================================================
# CONFIGURATION - CHOOSE YOUR SPEED OPTIMIZATION LEVEL
# ====================================================================
OPTIMIZATION_LEVEL = "FAST"  # Options: "FAST", "ULTRA_FAST", "BALANCED"

# Speed configurations
CONFIGS = {
    "FAST": {
        "model": "distilroberta-base",
        "epochs": 3,
        "batch_size": 8,
        "max_len": 256,
        "freeze_layers": True,
        "freeze_percentage": 0.75,  # Freeze 50% of layers
        "filter_completeness": False,
        "min_completeness": 0.0,
        "description": "Fast training with frozen layers"
    },
    "ULTRA_FAST": {
        "model": "distilroberta-base",
        "epochs": 1,
        "batch_size": 8,
        "max_len": 256,
        "freeze_layers": True,
        "freeze_percentage": 0.7,  # Freeze 70% of layers
        "filter_completeness": True,
        "min_completeness": 0.5,  # Only use medium/high completeness resumes
        "description": "Ultra-fast training - only classifier head trained"
    },
    "BALANCED": {
        "model": "distilroberta-base",
        "epochs": 2,
        "batch_size": 6,
        "max_len": 384,
        "freeze_layers": True,
        "freeze_percentage": 0.3,  # Freeze 30% of layers
        "filter_completeness": False,
        "min_completeness": 0.0,
        "description": "Balanced speed vs accuracy"
    }
}

config = CONFIGS[OPTIMIZATION_LEVEL]
print("\n" + "="*70)
print(f"OPTIMIZATION LEVEL: {OPTIMIZATION_LEVEL}")
print(f"Description: {config['description']}")
print("="*70)
print(f"Model: {config['model']}")
print(f"Epochs: {config['epochs']}")
print(f"Batch Size: {config['batch_size']}")
print(f"Max Length: {config['max_len']}")
print(f"Freeze Layers: {config['freeze_layers']} ({int(config['freeze_percentage']*100)}% frozen)")
print(f"Filter by Completeness: {config['filter_completeness']} (min: {config['min_completeness']})")
print("="*70 + "\n")

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("ResumeScreener") \
    .master("local[2]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.default.parallelism", "4") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

print(f"Spark Version: {spark.version}")

# Load data
file_path = "file:///home/DasigiPrasanna/Documents/PYSPARK/final_removed_columns.csv"
print(f"\nLoading data from: {file_path}")
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

# Convert completeness_score to float (in case it was read as string)
from pyspark.sql.types import FloatType
df_spark = df_spark.withColumn("completeness_score", col("completeness_score").cast(FloatType()))

# Filter by completeness if enabled
if config['filter_completeness']:
    original_count = df_spark.count()
    df_spark = df_spark.filter(col("completeness_score") >= config['min_completeness'])
    filtered_count = df_spark.count()
    print(f"\n⚡ Filtered dataset: {original_count} → {filtered_count} rows (removed {original_count - filtered_count} low-completeness resumes)")

print(f"\nDataset: {df_spark.count()} rows, {len(df_spark.columns)} columns")

# Define UDF for combining resume text
def combine_resume_text_with_completeness(completeness_score, education, experience, 
                                         certifications, achievements, languages, interests):
    text_parts = []
    
    # Convert to float if it's a string
    try:
        completeness_score = float(completeness_score)
    except (ValueError, TypeError):
        completeness_score = 0.0
    
    if completeness_score >= 0.7:
        completeness_level = "high"
    elif completeness_score >= 0.5:
        completeness_level = "medium"
    else:
        completeness_level = "low"
    
    text_parts.append(f"Profile completeness: {completeness_level}")
    
    if education and str(education).strip() and str(education) != 'None':
        text_parts.append(f"Education: {education}")
    
    if experience and str(experience).strip() and str(experience) != 'None':
        text_parts.append(f"Experience: {experience}")
    
    if certifications and str(certifications).strip() and str(certifications) != 'None':
        text_parts.append(f"Certifications: {certifications}")
    
    if achievements and str(achievements).strip() and str(achievements) != 'None':
        text_parts.append(f"Achievements: {achievements}")
    
    if languages and str(languages).strip() and str(languages) != 'None':
        text_parts.append(f"Languages: {languages}")
    
    if interests and str(interests).strip() and str(interests) != 'None':
        text_parts.append(f"Interests: {interests}")
    
    return " ".join(text_parts)

combine_text_udf = udf(combine_resume_text_with_completeness, StringType())

# Apply text combination
print("\nCombining resume fields...")
df_spark = df_spark.withColumn(
    "combined_text",
    combine_text_udf(
        col("completeness_score"),
        col("Education"),
        col("Experience"),
        col("Certifications"),
        col("Achievements"),
        col("Languages"),
        col("Interests")
    )
)

# Create completeness bins
df_spark = df_spark.withColumn(
    "completeness_bin",
    when(col("completeness_score") < 0.5, "Low")
    .when((col("completeness_score") >= 0.5) & (col("completeness_score") < 0.7), "Medium")
    .otherwise("High")
)

print("\nCompleteness Distribution:")
df_spark.groupBy("completeness_bin").count().orderBy("completeness_bin").show()

# Convert to Pandas
print("\nConverting to Pandas for train/test split...")
df_pandas = df_spark.select("combined_text", "Category", "completeness_score", "completeness_bin").toPandas()

# Label encoding
label_encoder = LabelEncoder()
df_pandas['encoded_labels'] = label_encoder.fit_transform(df_pandas['Category'])

unique_categories = df_pandas['Category'].unique()
num_labels = len(unique_categories)
print(f"\nCategories: {unique_categories}")
print(f"Number of labels: {num_labels}")

# Create stratification variable
df_pandas['strat_variable'] = df_pandas['Category'].astype(str) + '_' + df_pandas['completeness_bin'].astype(str)

# =============================================================
# SAFE STRATIFIED SPLIT
# =============================================================
from collections import Counter

# Count each strat group
counts = Counter(df_pandas["strat_variable"])
too_small = [grp for grp, c in counts.items() if c < 2]

if too_small:
    print(f"\n⚠️ Removing {len(too_small)} rare strat groups (<2 samples):")
    print(too_small[:10], "..." if len(too_small) > 10 else "")
    df_pandas = df_pandas[~df_pandas["strat_variable"].isin(too_small)]

# First split → train vs temp (val+test)
train_df, temp_df = train_test_split(
    df_pandas,
    test_size=0.2,
    random_state=SEED,
    stratify=df_pandas["strat_variable"]
)

# Second split → val vs test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=SEED,
    stratify=temp_df["strat_variable"]
)

# Unpack for consistency
train_texts, train_labels, train_completeness = (
    train_df["combined_text"].tolist(),
    train_df["encoded_labels"].tolist(),
    train_df["completeness_score"].tolist()
)

val_texts, val_labels, val_completeness = (
    val_df["combined_text"].tolist(),
    val_df["encoded_labels"].tolist(),
    val_df["completeness_score"].tolist()
)

test_texts, test_labels, test_completeness = (
    test_df["combined_text"].tolist(),
    test_df["encoded_labels"].tolist(),
    test_df["completeness_score"].tolist()
)

print(f"\nDataset splits:")
print(f"Train: {len(train_texts)} samples")
print(f"Validation: {len(val_texts)} samples")
print(f"Test: {len(test_texts)} samples")


print(f"\nDataset splits:")
print(f"Train: {len(train_texts)} samples")
print(f"Validation: {len(val_texts)} samples")
print(f"Test: {len(test_texts)} samples")

print("\n" + "="*70)
print("DATA PREPARATION COMPLETED")
print("="*70)

# Import PyTorch/Transformers
print("\nLoading PyTorch and Transformers...")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Model setup
MODEL_NAME = config['model']
print(f"\nLoading {MODEL_NAME} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

BATCH_SIZE = config['batch_size']
MAX_LEN = config['max_len']
print(f"Batch size: {BATCH_SIZE}")
print(f"Max length: {MAX_LEN}")

# Create datasets
print("\nCreating datasets...")
train_dataset = ResumeDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
val_dataset = ResumeDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)
test_dataset = ResumeDataset(test_texts, test_labels, tokenizer, max_len=MAX_LEN)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Initialize model
print(f"\nLoading {MODEL_NAME} model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels
)

# ====================================================================
# TRANSFER LEARNING: FREEZE LAYERS
# ====================================================================
if config['freeze_layers']:
    print(f"\n⚡ Applying transfer learning: Freezing {int(config['freeze_percentage']*100)}% of model layers")
    
    # Get all parameters
    all_params = list(model.named_parameters())
    total_params = len(all_params)
    freeze_count = int(total_params * config['freeze_percentage'])
    
    # Freeze the first N% of layers (bottom layers)
    frozen_count = 0
    trainable_count = 0
    
    for idx, (name, param) in enumerate(all_params):
        if idx < freeze_count:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1
    
    print(f"  ✓ Frozen parameters: {frozen_count}")
    print(f"  ✓ Trainable parameters: {trainable_count}")
    print(f"  ✓ Training only top {100-int(config['freeze_percentage']*100)}% of layers + classifier head")
    
    # Show which layers are trainable
    print("\n  Trainable layers:")
    for name, param in all_params:
        if param.requires_grad:
            print(f"    - {name}")
            if len([n for n, p in all_params if p.requires_grad]) > 10:
                print(f"    ... and {trainable_count - 10} more layers")
                break
else:
    print("\n⚠ Training all layers (no freezing)")

model.to(device)

# Optimizer - only for trainable parameters
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

EPOCHS = config['epochs']
num_training_steps = len(train_loader) * EPOCHS
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

print(f"\nTraining configuration:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Learning rate: 2e-5")
print(f"  - Training steps: {num_training_steps}")

# Training function
def train_one_epoch(model, loader, optimizer, device, scheduler):
    model.train()
    running_loss = 0.0
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        
        if batch_idx % 50 == 0:
            elapsed = time.time() - epoch_start
            batches_remaining = len(loader) - batch_idx
            time_per_batch = elapsed / (batch_idx + 1)
            eta_seconds = batches_remaining * time_per_batch
            eta_minutes = eta_seconds / 60
            
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}, ETA: {eta_minutes:.1f}min")
    
    return running_loss / max(1, len(loader))

# Evaluation function
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        
        preds.extend(pred.cpu().numpy().tolist())
        labs.extend(labels.cpu().numpy().tolist())
    
    return np.array(preds), np.array(labs)

# Performance analysis
def analyze_performance_by_completeness(predictions, true_labels, completeness_scores):
    print("\n=== PERFORMANCE BY COMPLETENESS LEVEL ===")
    
    completeness_bins = pd.cut(completeness_scores, 
                              bins=[0, 0.5, 0.7, 1.0], 
                              labels=['Low (<0.5)', 'Medium (0.5-0.7)', 'High (≥0.7)'],
                              include_lowest=True)
    
    for level in ['Low (<0.5)', 'Medium (0.5-0.7)', 'High (≥0.7)']:
        mask = completeness_bins == level
        if mask.sum() > 0:
            level_preds = predictions[mask]
            level_labels = true_labels[mask]
            
            acc = accuracy_score(level_labels, level_preds)
            f1 = f1_score(level_labels, level_preds, average='weighted')
            
            print(f"{level}: {mask.sum()} samples | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# Training loop
best_val_acc = 0.0
best_dir = f"resume_screener_{OPTIMIZATION_LEVEL.lower()}"

print("\n" + "="*70)
print("STARTING TRAINING...")
print("="*70)

total_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 70)
    
    epoch_start = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, device, scheduler)
    epoch_time = time.time() - epoch_start
    
    val_preds, val_labels_np = evaluate(model, val_loader, device)
    val_acc = accuracy_score(val_labels_np, val_preds)
    val_f1 = f1_score(val_labels_np, val_preds, average='weighted')
    
    print(f"\nEpoch {epoch} completed in {epoch_time/60:.1f} minutes")
    print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    analyze_performance_by_completeness(val_preds, val_labels_np, val_completeness)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"✓ New best model saved with validation accuracy: {val_acc:.4f}")

total_training_time = time.time() - total_start_time

# Final evaluation
print("\n" + "="*70)
print("LOADING BEST MODEL FOR FINAL EVALUATION...")
print("="*70)

best_model = AutoModelForSequenceClassification.from_pretrained(best_dir)
best_model.to(device)

test_preds, test_labels_np = evaluate(best_model, test_loader, device)

# Calculate metrics
acc = accuracy_score(test_labels_np, test_preds)
prec = precision_score(test_labels_np, test_preds, average='weighted')
rec = recall_score(test_labels_np, test_preds, average='weighted')
f1 = f1_score(test_labels_np, test_preds, average='weighted')

print("\n=== FINAL TEST EVALUATION ===")
print(f"Overall Accuracy : {acc:.4f}")
print(f"Overall Precision: {prec:.4f}")
print(f"Overall Recall   : {rec:.4f}")
print(f"Overall F1-score : {f1:.4f}")

analyze_performance_by_completeness(test_preds, test_labels_np, test_completeness)

print("\nClassification Report:")
print(classification_report(test_labels_np, test_preds, target_names=unique_categories))

# Save artifacts
print(f"\nSaving artifacts to {best_dir}/...")
joblib.dump(label_encoder, os.path.join(best_dir, 'label_encoder.pkl'))

completeness_analysis = {
    'test_completeness': test_completeness,
    'test_predictions': test_preds,
    'test_labels': test_labels_np
}
joblib.dump(completeness_analysis, os.path.join(best_dir, 'completeness_analysis.pkl'))

# Save configuration
config_info = {
    'optimization_level': OPTIMIZATION_LEVEL,
    'config': config,
    'training_time_minutes': total_training_time / 60,
    'final_accuracy': acc,
    'final_f1': f1
}
joblib.dump(config_info, os.path.join(best_dir, 'training_config.pkl'))

print(f"✓ All artifacts saved to: {best_dir}")

# Stop Spark
spark.stop()

# Final summary
print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)
print(f"Optimization Level: {OPTIMIZATION_LEVEL}")
print(f"Total Training Time: {total_training_time/60:.1f} minutes ({total_training_time/3600:.2f} hours)")
print(f"Model: {config['model']}")
print(f"Frozen Layers: {int(config['freeze_percentage']*100)}%")
print(f"Final Test Accuracy: {acc:.4f}")
print(f"Final Test F1: {f1:.4f}")
print("="*70)