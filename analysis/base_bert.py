#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!nvidia-smi


# In[2]:


from utils.cloudant_utils import cloudant_db as db
import numpy as np
import pandas as pd


# In[3]:


repos = [r for r in db.get_query_result({"type": "release"}, ["_id", "releases"], limit=10000, raw_result=True)["docs"]]
repos[0]['releases'][0].keys()


# In[4]:


values = [r for release in repos for r in release["releases"]]
df = pd.DataFrame(values)
df['contributors'] = df['contributors'].apply(lambda x:
                                              [i for i in x if i is not None] if isinstance(x, list)
                                              else [])


# In[114]:


df.shape


# In[116]:


df = df[~df['readme'].isnull()]


# In[117]:


new_df = df.groupby("repo").agg({"readme": list,
                                 "stars": sum,
                                 "forks": sum,
                                 "downloads": sum,
                                 "contributors": sum})


# In[118]:


readme1 =  new_df.iloc[2]['readme'][0]
readme2 = new_df.iloc[2]['readme'][1]
# print(len(new_df))
# new_df[new_df['readme'].map(lambda d: len(d)) > 0]


# In[119]:


new_df['600stars']= np.where(new_df['stars'] > 600, 1, 0)


# In[120]:


new_df.loc[new_df['600stars'] == 1].sample(5)[['readme', '600stars']]


# In[ ]:


# max_len = 0

#for sentences in readme:
#    if sentences:
#        for sent in sentences:
#            if sent:
#                input_ids = tokenizer.encode(sent, add_special_tokens=True)
#                max_len = max(max_len, len(input_ids))
#print('Max sentence length: ', max_len)


# In[9]:


import difflib
def diff_calculator(str1, str2):
   s = difflib.SequenceMatcher(lambda x : x == '')
   s.set_seqs(str1, str2)
   i = 1
   codes = []
   delete = []
   replace = {}
   insert = []
   for (opcode, before_start, before_end, after_start, after_end) in s.get_opcodes():
       if opcode == 'equal':
           continue
       codes.append(opcode)
       # print (i, ". %7s '%s :'  ----->  '%s'" % (opcode, test[0][before_start:before_end], test[1][after_start:after_end]))
       if opcode == 'replace':
           replace[str1[before_start:before_end]]  = str2[after_start:after_end]
       if opcode == 'delete':
           delete.append(str1[before_start:before_end])
       if opcode == 'insert':
           insert.append(str2[after_start:after_end])
       i = i + 1
   return replace, delete, insert


# In[15]:


# import re
# def clean(str):
#    return re.sub('\s+', ' ', str) if str is not None else ''
# i = 1
# replace, _, _ = diff_calculator('haha', "HAHA")
# for e in replace.keys():
#    print(i, '. ', clean(e), ' -->', clean(replace[e]))
#    i = i + 1
   # print(e, ' -->', (replace[e]))


# In[21]:


def create_a_sequence(readmeList):
   result = []
   for i in range(0,len(readmeList)-1):
       first = readmeList[i]
       second = readmeList[i+1]
       _, _, insert = diff_calculator(first, second)
       result.append(','.join(insert))
   return result


# In[22]:


# new_df['readme_diff'] = new_df['readme'].apply(lambda x: create_a_sequence(x))


# In[23]:


result = create_a_sequence(new_df.iloc[2]['readme'])
result[1]


# In[28]:


def prepareSequenceForBERT(readmeList):
    diffList = create_a_sequence(readmeList)
    s = '[CLS]' + "SEP".join([str(i) for i in diffList])
    return s


# In[122]:


from sklearn.model_selection import train_test_split
import torch
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df_train, df_test = train_test_split(new_df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
df_train.shape, df_val.shape, df_test.shape


# In[97]:


from torch.utils.data import Dataset, DataLoader
class ReadmeDataSet(Dataset):
   def __init__(self, df, tokenizer, max_len):
      self.df = df
      self.tokenizer = tokenizer
      self.max_len = max_len

   def __len__(self):
      return len(self.df)

   def __getitem__(self, item):
      sequence = prepareSequenceForBERT(self.df.iloc[item]['readme'][0])
      # readmes = self.df.iloc[item]['readme']
      # for i in range(0,len(readmes)-1):
      #   first = readmes[i]
      #   second = readmes[i+1]
      #   _, _, insert = diff_calculator(first, second)
      #   seq = ','.join(insert)
      #
      # sequence = '[CLS]' + "SEP".join([str(i) for i in diffList])

      # sequence = "Hi"
      target = self.df.iloc[item]['600stars']
      encoding = self.tokenizer.encode_plus(sequence,
                                     None,
                                     max_length = self.max_len,
                                     truncation=True,
                                     add_special_tokens=True,
#                                      padding=MAX_LEN,
#                                      padding='longest',
                                     pad_to_max_length=True,
                                     return_token_type_ids=True)

      return {
      'sequence': sequence,
      'input_ids': torch.tensor(encoding.input_ids, dtype=torch.long),
      'attention_mask':  torch.tensor(encoding.attention_mask, dtype=torch.long),
      'token_type_ids': torch.tensor(encoding.token_type_ids, dtype=torch.long),
      'targets': torch.tensor(target, dtype=torch.long)
      }


# In[98]:


def create_data_loader(df, tokenizer, max_len, batch_size):
   ds = ReadmeDataSet(
      df = df,
      tokenizer=tokenizer,
      max_len=max_len
   )
   return DataLoader(
      ds,
      batch_size=batch_size,
      num_workers=0
  )


# Import BERT Tokenizer and BERT Model

# In[82]:


from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=2,
                                                      output_attentions= False,
                                                      output_hidden_states= False)


# In[123]:


BATCH_SIZE = 2 #16
MAX_LEN = 100
train_data_loader = create_data_loader(df_train, bert_tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, bert_tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, bert_tokenizer, MAX_LEN, BATCH_SIZE)


# In[100]:


# for d in train_data_loader:
#     input_ids = d["input_ids"].to(device)
#     attention_mask = d["attention_mask"].to(device)
#     targets = d["targets"].to(device)


# In[101]:


#TEST the tokenizer and data loader
# sequence = prepareSequenceForBERT(new_df.iloc[2]['readme'])
# label = new_df.iloc[2]['600stars']
# tokens = bert_tokenizer.encode_plus(
#             sequence,
#             None,
#             max_length= 100,
#             truncation=True,
#             add_special_tokens=True,
# #             pad_to_max_length=True,
#             padding = True,
#             return_token_type_ids=True
#         )
# print(f' Sentence: {sequence}')
# print(f' Tokens: {tokens}')
# print(f' Tokens.token_type_ids: {tokens.token_type_ids}')
# print(f' Tokens.input_ids: {len(tokens.input_ids)}')
# output = {
#       'input_ids': torch.tensor(tokens.input_ids, dtype=torch.long),
#       'attention_mask':  torch.tensor(tokens.attention_mask, dtype=torch.long),
#       'token_type_ids': torch.tensor(tokens.token_type_ids, dtype=torch.long),
#       'targets': torch.tensor(label, dtype=torch.long)
#     }
# output


# In[102]:


# from torch import nn
# from transformers import BertModel
# class BertClassifier(nn.Module):
#     def __init__(self, dropout=0.5):
#         super(BertClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(768, 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, input_id, mask):
#         _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
#         dropout_output = self.dropout(pooled_output)
#         linear_output = self.linear(dropout_output)
#         final_layer = self.relu(linear_output)
#         return final_layer


# In[103]:


EPOCHS = 10
optimizer = AdamW(bert_model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
# loss_fn = nn.CrossEntropyLoss().to(device)


# In[104]:


from torch import nn
def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
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


# In[105]:


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


# In[124]:


get_ipython().run_cell_magic('time', '', "from collections import defaultdict\nhistory = defaultdict(list)\n\nloss_fn = nn.CrossEntropyLoss().to(device)\nbest_accuracy = 0\n\nfor epoch in range(EPOCHS):\n\n  print(f'Epoch {epoch + 1}/{EPOCHS}')\n  print('-' * 10)\n\n  train_acc, train_loss = train_epoch(\n    bert_model,\n    train_data_loader,\n    loss_fn,\n    optimizer,\n    device,\n    scheduler,\n    len(df_train)\n  )\n\n  print(f'Train loss {train_loss} accuracy {train_acc}')\n\n  val_acc, val_loss = eval_model(\n    bert_model,\n    val_data_loader,\n    loss_fn,\n    device,\n    len(df_val)\n  )\n\n  print(f'Val   loss {val_loss} accuracy {val_acc}')\n  print()\n\n  history['train_acc'].append(train_acc)\n  history['train_loss'].append(train_loss)\n  history['val_acc'].append(val_acc)\n  history['val_loss'].append(val_loss)\n\n  if val_acc > best_accuracy:\n    torch.save(bert_model.state_dict(), 'best_model_state.bin')\n    best_accuracy = val_acc")


# In[ ]:




