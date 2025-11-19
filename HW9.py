# ======================================================
# HW9：LLM 微調 - 情緒分類與憂鬱症風險監測（修正版：處理 LLaMA pad_token / collator / metrics）
# ======================================================

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
)
from peft import LoraConfig, get_peft_model
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # 可保留畫熱圖；若環境無此套件可改用純 matplotlib

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名稱:", torch.cuda.get_device_name(0))

# ======================================================
# 1️⃣ Dataset 載入與風險映射
# ======================================================
dataset = load_dataset("dair-ai/emotion")

label_names = dataset["train"].features["label"].names
# ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

emotion2risk = {
    "joy": 0, "love": 0, "surprise": 0,
    "anger": 1, "fear": 1,
    "sadness": 2
}

def map_risk(example):
    # 將數值 label 轉成情緒字串再映射到風險等級
    label_name = label_names[example["label"]]
    example["risk"] = emotion2risk[label_name]
    return example

dataset = dataset.map(map_risk)

# ======================================================
# 2️⃣ Zero-shot 推論（BART MNLI）
# ======================================================
zero_model = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=zero_model)
labels = ["joy", "love", "surprise", "anger", "fear", "sadness"]
sample_text = "I feel lonely and hopeless."
result = classifier(sample_text, candidate_labels=labels)
print("🔹 Zero-shot result:", result)

# ======================================================
# 3️⃣ Few-shot Prompt（示意）
# ======================================================
few_shot_examples = [
    ("I feel happy and relaxed", "joy"),
    ("I am terrified of losing my job", "fear"),
    ("I love spending time with my friends", "love"),
    ("I'm angry at how unfair things are", "anger"),
]
prompt = "\n".join([f"Text: {t}\nEmotion: {e}" for t, e in few_shot_examples])
prompt += "\nText: I feel empty and sad.\nEmotion:"
print("\n🧩 Few-shot Prompt:\n", prompt)

# ======================================================
# 4️⃣ LoRA 微調（TinyLlama as SEQ_CLS，補齊 pad_token）
# ======================================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# 對 decoder-only（如 LLaMA）補齊 pad_token，並用 EOS 作為 pad
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 建議右側 padding，避免與部分解碼器 cache 行為衝突
tokenizer.padding_side = "right"

num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    # 如需節省記憶體可開啟下兩行（視你的 CUDA 版本/硬體支援）
    # torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    # low_cpu_mem_usage=True,
)
# 確保模型知道 pad_token_id，否則 batch_size>1 會報錯
model.config.pad_token_id = tokenizer.pad_token_id
# 使用 decoder-only + Trainer 時建議關閉 cache
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# LoRA 設定：只套在 q_proj / v_proj
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config)

# ======================================================
# Tokenization
# ======================================================
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

# 保留 label 欄位，不要移除
tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text", "risk"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


train_ds = tokenized["train"]
val_ds   = tokenized["validation"]
test_ds  = tokenized["test"]

# 建議用 DataCollatorWithPadding，並讓它處理張量化
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# ======================================================
# 5️⃣ Metrics
# ======================================================
metric_f1  = evaluate.load("f1")
metric_auc = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]

    # ✅ 修正版：roc_auc 接收 predictions 而不是 prediction_scores
    try:
        auroc = metric_auc.compute(
            predictions=probs,  # 改這裡
            references=labels,
            average="macro",
            multi_class="ovr"
        )["roc_auc"]
    except Exception as e:
        print("⚠️ ROC-AUC 計算錯誤，已略過：", e)
        auroc = float("nan")

    return {"f1": f1, "auroc": auroc}

# ======================================================
# 6️⃣ TrainingArguments & Trainer
# ======================================================
bf16_ok = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs",    exist_ok=True)

args = TrainingArguments(
    output_dir="./outputs",
    do_train=True,
    do_eval=True,
    learning_rate=2e-4,
    num_train_epochs=2,
    per_device_train_batch_size=4,      # 如顯存吃緊可改 1~2
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,      # 等效更大 batch
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),  # 若要開啟 fp16 訓練（支援3090）

)

# 把模型搬到 GPU（若可用）
if torch.cuda.is_available():
    model = model.to("cuda")
    print("✅ 模型已移至 GPU：", torch.cuda.get_device_name(0))
else:
    print("⚠️ 使用 CPU 訓練，可能較慢。")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,  # 交由 collator 做動態 padding
    compute_metrics=compute_metrics,
)

trainer.train()

# ======================================================
# 7️⃣ 測試集評估 + 視覺化
# ======================================================
preds_output = trainer.predict(test_ds)
y_true = preds_output.label_ids
logits = preds_output.predictions[0] if isinstance(preds_output.predictions, (tuple, list)) else preds_output.predictions
y_pred = np.argmax(logits, axis=-1)

print("\n=== Classification Report (macro) ===")
print(classification_report(y_true, y_pred, target_names=labels))

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/confusion_matrix.png", dpi=300)
print("✅ 混淆矩陣圖已儲存：output/confusion_matrix.png")
plt.show()

# 憂鬱風險等級（0=低、1=中、2=高）
risk_preds = [emotion2risk[labels[p]] for p in y_pred]

# 走勢圖
plt.figure(figsize=(10, 4))
plt.plot(risk_preds, label="Predicted Risk (0=low, 1=mid, 2=high)")
plt.title("Depression Risk Trend")
plt.xlabel("Sample Index")
plt.ylabel("Risk Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/risk_trend.png", dpi=300)
print("✅ 風險走勢圖已儲存：output/risk_trend.png")
plt.show()

# 簡易熱圖（以行列重塑顯示連續視窗；若測試樣本不是 50 的倍數，可調整 reshape 參數）
h = 50
w = int(np.ceil(len(risk_preds) / h))
pad_len = h * w - len(risk_preds)
heat_vec = np.array(risk_preds + [np.nan] * pad_len)
heat_mat = heat_vec.reshape(h, -1)

plt.figure(figsize=(8, 4))
sns.heatmap(heat_mat, cmap="coolwarm", cbar=True, vmin=0, vmax=2)
plt.title("High Risk Heatmap (rolling view)")
plt.xlabel("Window")
plt.ylabel("Index (chunked)")
plt.tight_layout()
plt.savefig("output/risk_heatmap.png", dpi=300)
print("✅ 高風險熱圖已儲存：output/risk_heatmap.png")
plt.show()

# ======================================================
# 8️⃣ Precision–Recall Curve + PR-AUC
# ======================================================
from sklearn.metrics import precision_recall_curve, auc

# 取得 softmax 機率
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

# 我們對「高風險情緒 sadness」做 PR-AUC（若要 multi-class micro-average 我也可以幫你改）
# sadness 在 labels = ["joy","love","surprise","anger","fear","sadness"] 裡是 index=5
sadness_index = labels.index("sadness")

y_true_binary = (y_true == sadness_index).astype(int)       # 1 = sadness (高風險)
y_prob_binary = probs[:, sadness_index]                     # 該類別的機率

precision, recall, thresholds = precision_recall_curve(
    y_true_binary,
    y_prob_binary
)

pr_auc = auc(recall, precision)
print(f"PR-AUC (sadness / high-risk): {pr_auc:.4f}")

# --- 作圖 ---
plt.figure(figsize=(7, 6))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (High-risk: sadness)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output/pr_auc.png", dpi=300)
print("✅ PR-AUC 圖已儲存：output/pr_auc.png")
plt.show()
