
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ================================
# 1. 기본 설정 및 데이터 불러오기
# ================================
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")

# ================================
# 2. 모델 및 토크나이저 로딩
# ================================
model_name = "google/gemma-2b-it"
hf_token="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"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name, token=hf_token, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ================================
# 3. Prompt 포맷 정의
# ================================
def build_prompt(text):
    return (
        "You are a Korean teacher. Your job is to fix malformed or broken Korean letters and restore them into natural, correct Korean.\n\n"
        f"Input: {text}\nOutput:"
    )

# ================================
# 4. 데이터 전처리 및 토큰화
# ================================
class CorrectionDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=512):
        self.data = []
        for input_text, target_text in zip(inputs, targets):
            prompt = build_prompt(input_text)
            combined = prompt + " " + target_text
            encodings = tokenizer(
                combined,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"][0]
            attention_mask = encodings["attention_mask"][0]

            # prompt 길이 파악
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])

            labels = input_ids.clone()
            labels[:prompt_len] = -100  # prompt는 학습하지 않음

            self.data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = CorrectionDataset(train_df["input"].tolist(), train_df["output"].tolist(), tokenizer)

# ================================
# 5. 학습 설정
# ================================
training_args = TrainingArguments(
    output_dir="./gemma_correction",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2,
    save_steps=500,
    logging_steps=100,
    fp16=False  # 문제 방지 위해 일단 꺼둠
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# ================================
# 6. 학습 시작
# ================================
trainer.train()

# ================================
# 7. Inference 및 결과 저장
# ================================
submission_df = pd.read_csv("./sample_submission.csv")
test_inputs = test_df["input"].tolist()


def restore_with_prompt(sentence):
    prompt = build_prompt(sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Output:")[-1].strip()

predictions = [restore_with_prompt(text) for text in test_inputs]
submission_df["output"] = predictions
submission_df.to_csv("./submission_gemma.csv", index=False)

print("✅ 제출 파일 저장 완료: submission_gemma.csv")
