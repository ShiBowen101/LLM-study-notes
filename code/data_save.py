from datasets import load_dataset
import datasets as d
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertConfig, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import pandas as pd
import torch
from torch.utils.data import Dataset

ds_train = pd.read_parquet("train.parquet")
ds_test = pd.read_parquet("test.parquet")
# 随机打乱处理
ds_train = ds_train.sample(frac=1).reset_index(drop=True)
ds_test = ds_test.sample(frac=1).reset_index(drop=True)
ds_train.to_csv("train.csv")
ds_test.to_csv("test.csv")