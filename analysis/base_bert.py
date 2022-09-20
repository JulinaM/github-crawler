#!/usr/bin/env python
# coding: utf-8

#!nvidia-smi
import sys
sys.path.append('../')

BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 10
RANDOM_SEED = 42
model_name = 'bert-base-uncased'


from utils.cloudant_utils import cloudant_db as db
import numpy as np
import pandas as pd
from datetime import datetime
import difflib
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
from readme_cleanup import readme_cleanup
import random


repos = [r for r in db.get_query_result({"type": "release"}, ["_id", "releases"], limit=10000, raw_result=True)["docs"]]
values = [r for release in repos for r in release["releases"]]
df = pd.DataFrame(values)
df['contributors'] = df['contributors'].apply(lambda x:
                                              [i for i in x if i is not None] if isinstance(x, list)
                                              else [])
df = df[~df['readme'].isnull()]
new_df = df.groupby("repo").agg({"readme": list, "total_stars": list})
new_df = new_df[new_df['readme'].map(len) > 2]
new_df = new_df[:100]


def diff_calculator(str1, str2):
   s = difflib.SequenceMatcher(lambda x : x == '')
   s.set_seqs(str1, str2)
   i = 1
   # codes = []
   # delete = []
   # replace = {}
   insert = []
   for (opcode, before_start, before_end, after_start, after_end) in s.get_opcodes():
       if opcode == 'equal':
           continue
       # codes.append(opcode)
       # # print (i, ". %7s '%s :'  ----->  '%s'" % (opcode, test[0][before_start:before_end], test[1][after_start:after_end]))
       # if opcode == 'replace':
       #     replace[str1[before_start:before_end]]  = str2[after_start:after_end]
       # if opcode == 'delete':
       #     delete.append(str1[before_start:before_end])
       if opcode == 'insert':
           if str2[after_start:after_end]:
            insert.append(str2[after_start:after_end])
       i = i + 1
   # return replace, delete, insert
   return insert

def create_a_sequence(readmeList):
    result = []
    for i in range(0,len(readmeList)-1):
        first = readme_cleanup(readmeList[i])
        second = readme_cleanup(readmeList[i+1])
        insert = diff_calculator(first, second)
        result.append(','.join(insert))
    return result

def prepareSequenceForBERT(readmeList):
    diffList = create_a_sequence(readmeList)
    s = '[CLS]' + "[SEP]".join([str(i) for i in diffList])
    return s +'[SEP]'


new_df['sequence']= new_df['readme'].apply(prepareSequenceForBERT)


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

df_train, df_test = train_test_split(new_df, test_size=0.4, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
df_train.shape, df_val.shape, df_test.shape


class ReadmeDataSet(Dataset):
   def __init__(self, _df, tokenizer, max_len):
      self._df = _df
      self.tokenizer = tokenizer
      self.max_len = max_len

   def __len__(self):
      return len(self._df)

   def __getitem__(self, item):
      sequence = self._df.iloc[item]['sequence']
      total_stars = self._df.iloc[item]['total_stars']
    
      k = random.randint(2, len(total_stars))
      _sequence = sequence[:k]
      target_value = total_stars[k-1]
      target = 1 if target_value > 600 else 0
      encoding = self.tokenizer.encode_plus(_sequence,
                                     None,
                                     max_length = self.max_len,
                                     truncation=True,
                                     add_special_tokens=True,
                                     pad_to_max_length=True,
                                     return_token_type_ids=True)

      return {
      'sequence': _sequence,
      'input_ids': torch.tensor(encoding.input_ids, dtype=torch.long),
      'attention_mask':  torch.tensor(encoding.attention_mask, dtype=torch.long),
      'token_type_ids': torch.tensor(encoding.token_type_ids, dtype=torch.long),
      'targets': torch.tensor(target, dtype=torch.long)
      }


def create_data_loader(_df, tokenizer, max_len, batch_size):
   ds = ReadmeDataSet(_df = _df, tokenizer=tokenizer, max_len=max_len)
   return DataLoader(ds, batch_size=batch_size, num_workers=0)


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=2,
                                                      output_attentions= False,
                                                      output_hidden_states= False)


train_data_loader = create_data_loader(df_train, bert_tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, bert_tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, bert_tokenizer, MAX_LEN, BATCH_SIZE)
new_df.shape, df_train.shape, df_val.shape, df_test.shape


optimizer = AdamW(bert_model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def train_epoch( model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

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

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs.logits, dim=1)

      loss = loss_fn(outputs.logits, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


get_ipython().run_cell_magic('time', '', 'history = defaultdict(list)\nloss_fn = nn.CrossEntropyLoss().to(device)\nbest_accuracy = 0\n\nfor epoch in range(EPOCHS):\n\n  print(f\'Epoch {epoch + 1}/{EPOCHS}\')\n  print(\'-\' * 10)\n\n  train_acc, train_loss = train_epoch(\n    bert_model,\n    train_data_loader,\n    loss_fn,\n    optimizer,\n    device,\n    scheduler,\n    len(df_train)\n  )\n\n  print(f\'Train loss {train_loss} accuracy {train_acc}\')\n\n  val_acc, val_loss = eval_model(\n    bert_model,\n    val_data_loader,\n    loss_fn,\n    device,\n    len(df_val)\n  )\n\n  print(f\'Val   loss {val_loss} accuracy {val_acc}\')\n  print()\n\n  history[\'train_acc\'].append(train_acc)\n  history[\'train_loss\'].append(train_loss)\n  history[\'val_acc\'].append(val_acc)\n  history[\'val_loss\'].append(val_loss)\n\n  if val_acc > best_accuracy:\n    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M%p")\n    torch.save(bert_model.state_dict(), \'best_model_state\' + current_time+\'.bin\')\n    best_accuracy = val_acc\n')


test_acc, _ = eval_model(bert_model, test_data_loader, loss_fn, device, len(df_test))
test_acc.item()


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


outputss, y_sequences, y_pred, y_pred_probs, y_test = get_predictions(bert_model, train_data_loader)


logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test.numpy(), y_pred_probs[:, 1:].numpy())
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_bert_ROC_'+ current_time +'.png')
plt.show()


y_pred_probs_pd = [y.numpy() for y in y_pred_probs]
someListOfLists = list(zip(y_sequences, y_test.numpy(), y_pred.numpy(), y_pred_probs[:, 1:].numpy().squeeze(), y_pred_probs_pd ))
npa = np.asarray(someListOfLists)
dff = pd.DataFrame(someListOfLists, columns = ['readme', 'Real', 'Predicted', 'Pred-prob', 'All Pred-probs' ])
print(dff)

