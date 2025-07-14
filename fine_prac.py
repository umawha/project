from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# 모델과 토크나이저 로딩
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Gemma는 LLaMA 계열이라 보통 이 부분
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# 예제용 데이터셋 사용 (Hugging Face 공개 데이터셋 중 하나)
dataset = load_dataset("Abirate/english_quotes")  # 아주 작은 텍스트 데이터셋

def preprocess(example):
    prompt = f"### Instruction:\n{example['quote']}\n### Response:\n"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset["train"].map(preprocess)

# 훈련 인자
training_args = TrainingArguments(
    output_dir="./gemma2b-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
