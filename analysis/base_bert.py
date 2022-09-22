#!/usr/bin/env python
# coding: utf-8

#!nvidia-smi
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch import nn
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from prepare_sequence import prepareSequenceForBERT
import random


BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 16
RANDOM_SEED = 42
model_name = 'bert-base-uncased'
threshold = 1

def load_df(filepath):
  if filepath:
    try:
      print('Reading from ', filepath)
      new_df = pd.read_csv('csv/inputForBERT_2022_09_21-09_14PM.csv')
    except:
      print('Failed to load ', filepath)

  else:
    from utils.cloudant_utils import cloudant_db as db
    repos = [r for r in db.get_query_result({"type": "release"}, ["_id", "releases"], limit=10000, raw_result=True)["docs"]]
    values = [r for release in repos for r in release["releases"]]
    df = pd.DataFrame(values)
    df['contributors'] = df['contributors'].apply(lambda x:
                                                  [i for i in x if i is not None] if isinstance(x, list)
                                                  else [])
    df = df[~df['readme'].isnull()]
    new_df = df.groupby("repo").agg({"readme": list, "total_stars": list})
    new_df = new_df[new_df['readme'].map(len) > threshold]
#     new_df = new_df[:10]

    new_df['k'] = new_df['total_stars'].map(lambda x: random.randint(threshold, len(x)))
    new_df['readme1'] = new_df.apply(lambda x: x.readme[:x.k], axis=1)
    new_df['target_val'] = new_df.apply(lambda x: x.total_stars[x.k-1], axis=1)
    new_df['target'] = new_df.apply(lambda x: 1 if x.target_val> 600 else 0, axis=1)
    print(new_df['target'].value_counts())

    print(40*"*", 'Preparing sequence for BERT')
    new_df['sequence']= new_df['readme1'].apply(prepareSequenceForBERT)

    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M%p")
    filepath = 'csv/inputForBERT_' + current_time + '.csv'
    new_df.to_csv(filepath)
    print('Sequence File saved to ', filepath)

  return new_df


class ReadmeDataSet(Dataset):
   def __init__(self, _df, tokenizer, max_len):
      self._df = _df
      self.tokenizer = tokenizer
      self.max_len = max_len

   def __len__(self):
      return len(self._df)

   def __getitem__(self, item):
      sequence = self._df.iloc[item]['sequence']
      target = self._df.iloc[item]['target']      

      encoding = self.tokenizer.encode_plus(sequence,
                                     None,
                                     max_length = self.max_len,
                                     truncation=True,
                                     add_special_tokens=True,
                                     pad_to_max_length=True,
                                     return_token_type_ids=True)

      return {
      'sequence': sequence,
      'input_ids': torch.tensor(encoding.input_ids, dtype=torch.long),
      'attention_mask':  torch.tensor(encoding.attention_mask, dtype=torch.long),
      'token_type_ids': torch.tensor(encoding.token_type_ids, dtype=torch.long),
      'targets': torch.tensor(target, dtype=torch.long)
      }


def create_data_loader(_df, tokenizer, max_len, batch_size):
   ds = ReadmeDataSet(_df = _df, tokenizer=tokenizer, max_len=max_len)
   return DataLoader(ds, batch_size=batch_size, num_workers=0)


def train_epoch( model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    _, preds = torch.max(outputs.logits, dim=1)
    loss = loss_fn(outputs.logits, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs.logits, dim=1)

      loss = loss_fn(outputs.logits, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
  model = model.eval()

  sequences = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["sequence"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      token_type_ids = d["token_type_ids"].to(device)
      targets = d["targets"].to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

      _, preds = torch.max(outputs.logits, dim=1)
      probs = F.softmax(outputs.logits, dim=1)

      sequences.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return outputs, sequences, predictions, prediction_probs, real_values


if __name__ == "__main__":
    new_df = load_df(True)

    df_train, df_test = train_test_split(new_df, test_size=0.4, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    print(df_train.shape, df_val.shape, df_test.shape)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                num_labels=2,
                                                output_attentions= False,
                                                output_hidden_states= False)
    train_data_loader = create_data_loader(df_train, bert_tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, bert_tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, bert_tokenizer, MAX_LEN, BATCH_SIZE)
    print(new_df.shape, df_train.shape, df_val.shape, df_test.shape)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bert_model = bert_model.to(device)
    optimizer = AdamW(bert_model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print(40*"*", 'Training')
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    history = defaultdict(list)
    loss_fn = nn.CrossEntropyLoss().to(device)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(bert_model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model( bert_model, val_data_loader, loss_fn, device, len(df_val) )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
          current_time = datetime.now().strftime("%Y_%m_%d-%I_%M%p")
          torch.save(bert_model.state_dict(), 'checkpoint/best_model_state' + current_time+'.bin')
          best_accuracy = val_acc

    test_acc, _ = eval_model(bert_model, test_data_loader, loss_fn, device, len(df_test))
    test_acc.item()

    outputss, y_sequences, y_pred, y_pred_probs, y_test = get_predictions(bert_model, train_data_loader)
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test.numpy(), y_pred_probs[:, 1:].numpy())
    plt.figure()
    plt.plot(fpr, tpr, label='BERT(area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('images/Log_bert_ROC_'+ current_time +'.png')
    plt.show()

    y_pred_probs_pd = [y.numpy() for y in y_pred_probs]
    someListOfLists = list(zip(y_sequences, y_test.numpy(), y_pred.numpy(), y_pred_probs[:, 1:].numpy().squeeze(), y_pred_probs_pd ))
    npa = np.asarray(someListOfLists)
    dff = pd.DataFrame(someListOfLists, columns = ['readme', 'Real', 'Predicted', 'Pred-prob', 'All Pred-probs' ])
    print(dff)
    dff.to_csv('csv/test_result' + current_time + '.csv')
