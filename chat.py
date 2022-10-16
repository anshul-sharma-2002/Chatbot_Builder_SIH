import json
import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
from transformers import TFDistilBertForSequenceClassification,DistilBertTokenizer
from collections import defaultdict
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

f = open("intents.json")
data = json.load(f)

lb = []
text = []
dt = data['intents']
out_len = len(dt)
for i in dt:
    for j in i['patterns']:
        text.append(j)
        lb.append(i['tag'])
        
df_data = {'text': text, 'label': lb}

df = pd.DataFrame.from_dict(df_data)

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

train,test = train_test_split(df,test_size=0.3,stratify=df['label'])

train['label'] =  pd.to_numeric(train['label'])
test['label'] =  pd.to_numeric(test['label'])

train_text, train_labels = train['text'], train['label']

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',use_fast=True)

def create_dataset(df):
    tokens = defaultdict(list)
    for i in df['text']:
        t = tokenizer(i, truncation=True,max_length=64,padding='max_length')
        tokens['input_ids'].append(t['input_ids'])
        tokens['attention_mask'].append(t['attention_mask'])
        tokens['labels']=df['label']
    tokens['labels']=tf.convert_to_tensor([tokens['labels']])
    tokens['labels'] = tf.reshape(tokens['labels'],(len(df),1))
    return tokens

train_tokens = create_dataset(train)
valid_tokens = create_dataset(test)

train_dataset = tf.data.Dataset.from_tensor_slices(dict(train_tokens))
train_dataset = train_dataset.batch(16)
valid_dataset = tf.data.Dataset.from_tensor_slices(dict(valid_tokens))
valid_dataset = valid_dataset.batch(16)

model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=len(le.classes_))

EPOCHS = 15
TRAINING_STEPS = len(train_dataset) * EPOCHS

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-4, 
    end_learning_rate=5e-5,
    decay_steps=TRAINING_STEPS
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=tf.metrics.SparseCategoricalAccuracy()
)

model.load_weights("./weights")

def predict_proba(text_list, model, tokenizer):  
    #tokenize the text
    encodings = tokenizer(text_list, 
                          max_length=64, 
                          truncation=True, 
                          padding=True)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
    preds = model.predict(dataset.batch(1)).logits  
    res = tf.nn.softmax(preds, axis=1).numpy()  
    res = tf.argmax(res,axis=1)
    intent = le.inverse_transform(res)
    
    for i in data['intents']: 
        
        if i["tag"] == intent[0]:
            result = ' '.join(i["responses"])
            break
    return result

def get_response(text):
    return predict_proba([text], model, tokenizer)

# print(get_response("treatment of cuts"))
    