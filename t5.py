from transformers import T5TokenizerFast, T5ForConditionalGeneration
print("Transformers 라이브러리에서 T5 모델과 토크나이저를 불러옵니다.")
import pandas as pd
train_df=pd.read_csv('./train.csv')
test_df=pd.read_csv('./test.csv')

#모델 다운로드
model_name = "paust/pko-t5-large"
tokenizer = T5TokenizerFast.from_pretrained(model_name, use_auth_token=False)
model = T5ForConditionalGeneration.from_pretrained(model_name, use_auth_token=False)



train_texts=['복원하세요: '+ text for text in train_df['input'].tolist()]
train_labels=train_df["output"].tolist()

# ✅ 3. 문장을 토큰화 (숫자로 변환)
def encode_texts(texts, labels, tokenizer, max_length=128):
    inputs = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    outputs = tokenizer(labels, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"]
    }

# ✅ 4. train 데이터를 토큰화하여 변환
train_data = encode_texts(train_texts, train_labels, tokenizer)

import torch
from torch.utils.data import Dataset, DataLoader

class T5Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

# ✅ 5. PyTorch Dataset 생성
train_dataset = T5Dataset(train_data)
# ✅ 6. PyTorch DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

from transformers import get_scheduler
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = len(train_loader) * 5
lr_scheduler = get_scheduler(
    "cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

from transformers import T5ForConditionalGeneration

# ✅ 7. T5 모델 불러오기
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./t5_finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_steps=500,
    logging_dir="./logs",
    eval_strategy="no",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(optimizer, lr_scheduler)
)
#토큰 : c8bcc751c0b9352a09d4a2011d4a28107fa23c75
# ✅ 8. 모델 학습 시작
trainer.train()

# ✅ sample_submission.csv 불러오기
submission_df = pd.read_csv("./sample_submission.csv")

def restore_sentence(model, tokenizer, sentence):
    model.eval()
    input_text = "복원하세요: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=50,
            num_beams=8,  # Beam Search 적용 (8개의 후보 중 최적의 문장 선택)
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ✅ test.csv에 있는 손상된 문장 예측
test_inputs = submission_df["ID"].tolist()  # sample_submission.csv에서 ID 가져오기
test_sentences = pd.read_csv("./test.csv")["input"].tolist()  # test.csv에서 input 가져오기

predictions = [restore_sentence(model, tokenizer, sent) for sent in test_sentences]

# ✅ 예측된 output을 sample_submission.csv 형식에 맞게 저장
submission_df["output"] = predictions

# ✅ 최종 제출 파일 저장
submission_df.to_csv("./submission_final.csv", index=False)

print("제출 파일 저장 완료: ./sample_submission.csv")


'''pip install tf-keras
pip install transformers
pip install torch
pip install pandas
pip install datasets
pip install transformers[torch]
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'''