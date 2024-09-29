from datasets import load_dataset
import datasets as d
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import BertConfig, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam


class TrainDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_parquet('train.parquet')

    def __getitem__(self, index):
        return self.data.iloc[index]["text"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_parquet('test.parquet')

    def __getitem__(self, index):
        return self.data.iloc[index]["text"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


ds_train = TrainDataset()
ds_test = TestDataset()
model_path = "GPT2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def collate_fn(batch):
    texts = []
    labels = []
    for one in batch:
        texts.append(one[0])
        labels.append(one[1])
    inputs = tokenizer(texts, max_length=128, padding=True, truncation=True, return_tensors="pt")

    inputs["labels"] = torch.tensor(labels)
    return inputs


train_loader = DataLoader(ds_train, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(ds_test, batch_size=64, collate_fn=collate_fn)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model = model.cuda()
model.config.pad_token_id = tokenizer.pad_token_id
optimizer = Adam(model.parameters(), lr=2e-5)


## 训练
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in test_loader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            prediction = torch.argmax(output.logits, dim=-1)
            acc_num += (prediction.long() == batch["labels"].long()).float().sum()
    return acc_num / len(ds_test)


def train(epoch=3, log_step=10):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in train_loader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
                optimizer.zero_grad()
                output = model(**batch)
                output.loss.backward()
                optimizer.step()
                print(global_step, log_step)
                if global_step % log_step == 0:
                    print(f"ep:{ep},global_step:{global_step},loss:{output.loss.item()}")
                global_step += 1
            acc = evaluate()
            print(f"ep:{ep},acc:{acc}")


before_acc = evaluate()
print(f"before trainning:{before_acc}")
train()
