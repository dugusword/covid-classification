#!/usr/bin/python
import os
import argparse
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import csv, json

class CovidDataset(Dataset):

    def __init__(self, frame):
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

def load_dataset(csv_file, mode='train', test_size=0.2):
    frame = pd.read_csv(csv_file, encoding='latin-1')
    if mode == 'train':
        train_frame, test_frame = train_test_split(frame, test_size=test_size)
        train_set = CovidDataset(train_frame)
        test_set = CovidDataset(test_frame)
    else:
        train_set = None
        test_set = CovidDataset(frame)
    
    return {'train': train_set, 'test': test_set}

    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = \
        precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': np.mean(f1),
        'precision': np.mean(precision),
        'recall': np.mean(recall)
    }
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval', 'predict'])
    
    parser.add_argument('--import_model', default='bert-base-uncased')
    
    parser.add_argument('--dataset_path', default='.')
    parser.add_argument('--output_dir', default='.')
    
    args = parser.parse_args()
    ds_path = args.dataset_path
    dataset = load_dataset(ds_path, args.mode)

    training_args = TrainingArguments(
        output_dir='{}/checkpoints'.format(args.output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy='steps',
        logging_steps=500,
        logging_dir='{}/logs'.format(args.output_dir)
    )

    model = BertForSequenceClassification.from_pretrained(args.import_model,
                                                          num_labels=5)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'eval':
        trainer.evaluate()
    elif args.mode == 'predict':
        res = trainer.predict(dataset['test'])
        #for row in res.predictions:
        #    print(np.argmax(row))
        idx = np.argmax(res.predictions, 1)
        
        metrics = compute_metrics(res)
        with open(args.output_dir + '/score.json', 'w') as f:
            json.dump(metrics, f)
        
        label_map = { 0: 'Extremely Positive',
                      1: 'Positive',
                      2: 'Neutral',
                      3: 'Negative',
                      4: 'Extremely Negative' }
        
        labels = []
        for i in idx:
            labels.append([label_map[i]])
        
        with open(args.output_dir + '/result.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(labels)
        
