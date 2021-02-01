#!/usr/bin/python
import torch
from torch.nn import functional as F
from transformers import BertForSequenceClassification, AdamW, BertTokenizer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()
optimizer = AdamW(model.parameters(), lr=1e-5)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_batch = ['I like AMD.', 'AMD sucks']
encoding = tokenizer(text_batch, return_tensors='pt',
                     padding=True, truncation=True)

input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
labels = torch.tensor([[1, 0]])
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
