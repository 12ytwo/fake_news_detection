import pandas as pd
import numpy as np
import jieba
import random
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

# 固定随机种子
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(123)

# 读取训练数据
train_df = pd.read_csv("train.csv")
train_df = train_df.dropna()
train_df = shuffle(train_df)

# 简单中文分词清洗
stopwords = ['是','的','了','在','和','有','被','这','那','之','更','与','对于','并','我','他','她','它','我们','他们','她们','它们']
punc = r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'

def cleaning(text):
    cutwords = list(jieba.lcut_for_search(str(text)))
    final_cutwords = ''
    for word in cutwords:
        if word not in stopwords and word not in punc:
            final_cutwords += word + ' '
    return final_cutwords

train_df['title'] = train_df['title'].apply(cleaning)

# 准备数据
model_name = "bert-base-chinese"
max_length = 512
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

texts = train_df['title'].tolist()
labels = train_df['label'].tolist()

train_texts, valid_texts, train_labels, valid_labels = train_test_split(texts, labels, test_size=0.2)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
valid_dataset = NewsDataset(valid_encodings, valid_labels)

# 模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    logging_dir='./logs',
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    logging_steps=200,
    save_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 保存模型
model.save_pretrained('./cache/model')
tokenizer.save_pretrained('./cache/tokenizer')

# 推理函数：返回真实新闻的概率
def get_prob(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    return float(probs[0][1].cpu().detach().numpy())  # 概率为 label=1 (真实新闻)

# 读取测试集
test_df = pd.read_csv("test.csv")
test_df['title'] = test_df['title'].apply(cleaning)

# 预测
test_df['prob'] = test_df['title'].apply(get_prob)

# 保存结果
final_df = test_df[['id', 'prob']]
final_df.to_csv("result.csv", index=False)
