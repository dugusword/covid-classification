#!/usr/bin/python
from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_PATH = '/home/biye/covid-classification/results/checkpoint-7500'
label_map = { 0: 'Extremely Positive',
              1: 'Positive',
              2: 'Neutral',
              3: 'Negative',
              4: 'Extremely Negative' }

text = "Amazon recently restacked hand sanitizer. You can buy them very easily now. Check it out!"

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=5)

encoded = tokenizer(text, return_tensors='pt')
result = model(encoded['input_ids'], attention_mask=encoded['attention_mask'])
label_idx = result.logits.argmax()
label = label_map[int(label_idx)]
print(label)
