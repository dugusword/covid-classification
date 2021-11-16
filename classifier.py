#!/usr/bin/python
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class CovidDataset(Dataset):

    def __init__(self, csv_file):
        frame = pd.read_csv(csv_file, encoding='latin-1')
        self.frame = frame
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tweets = frame.iloc[:,4].tolist()
        self.tweets = tokenizer(self.tweets, return_tensors='pt', 
                                padding=True, truncation=True)
        self.labels = frame.iloc[:,5].tolist()
        
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        label_map = { 'Extremely Positive': torch.tensor(0),
                      'Positive': torch.tensor(1),
                      'Neutral': torch.tensor(2),
                      'Negative': torch.tensor(3),
                      'Extremely Negative': torch.tensor(4) }

        label = label_map[self.labels[idx]]
        ret = { 'label': label,
                'input_ids': self.tweets['input_ids'][idx],
                'attention_mask': self.tweets['attention_mask'][idx] }

        return ret


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = \
        precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': list(f1),
        'precision': list(precision),
        'recall': list(recall)
    }
    
if __name__ == '__main__':
    training_set = CovidDataset('Corona_NLP_train.csv')
    test_set = CovidDataset('Corona_NLP_test.csv')
    training_set[0]
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy='steps',
        logging_steps=500,
        logging_dir='./logs'
    )

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=5)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=training_set,
        eval_dataset=test_set
    )

    trainer.train()
    trainer.evaluate()
