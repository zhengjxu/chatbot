import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import string
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.layers import Input, Dense, LSTM, GRU, Embedding, TimeDistributed
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import ModelCheckpoint, LambdaCallback

DATA_NAME = 'gunthercox'
DATA_DIR_PATH = '../data/gunthercox/'
EMBEDDING_PATH = '../glove/glove.840B.300d.txt'
WEIGHT_FILE_PATH = f'../models/{DATA_NAME}/word-glove-weights.hdf5'
MODEL_FILE_PATH = f'../models/{DATA_NAME}/word-glove-model.h5'
MAX_FEATURES = 10_000
INPUT_MAX_LEN = 30
TARGET_MAX_LEN = 40
EMBEDDING_SIZE = 300
HIDDEN_UNITS = 128

def parse_q(text):
    text = text.lower()
    text = re.sub(r'\s', ' ', text)  # remove \n
    text = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', text).strip()
    text = ' '.join(text.split()[:INPUT_MAX_LEN])
    return text

def tokenize(x):
    # remove ',.!?<>' from filters
    filters='"#$%&()*+-/:;=@[\\]^_`{|}~\t\n'
    tk = Tokenizer(MAX_FEATURES, filters)
    tk.fit_on_texts(x)
    tokenized = tk.texts_to_sequences(x)
    return tokenized, tk

def padding(x, max_len):
    x = pad_sequences(x, max_len, padding='post')
    return x

def get_coef(embedding_path):
    embedding_index = {}
    with open(embedding_path) as f:
        for line in f:
            values = line.strip().split(' ')
            word = values[0]
            coef = np.array(values[1:], dtype='float32')
            embedding_index[word] = coef
    return embedding_index

def coef_matrix(vocab_size, embedding_size, word2idx):
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for w,i in word2idx.items():
        if i >= MAX_FEATURES:
            continue
        embedding_vector = embedding_index.get(w)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

print('load inputs')
with open(f'../models/{DATA_NAME}/proc_data.pickle', 'rb') as f:
    inputs, targets = pickle.load(f)

print('get Glove coef...')
embedding_index = get_coef(EMBEDDING_PATH)

print('get word index dict...')
with open(f'../models/{DATA_NAME}/word_index_dict.pickle', 'rb') as f:
    enc_tk, dec_tk = pickle.load(f)
input_vocab = min(MAX_FEATURES, len(enc_tk.word_index)+1)  # plus 1 for padding
target_vocab = min(MAX_FEATURES, len(dec_tk.word_index)+1)
word2idx = dec_tk.word_index
idx2word = {idx:w for w,idx in word2idx.items()}
idx2word[0] = '<pad>'

print('construct embedding matrix...')
input_embedding_matrix = coef_matrix(input_vocab, EMBEDDING_SIZE, enc_tk.word_index)
target_embedding_matrix = coef_matrix(target_vocab, EMBEDDING_SIZE, dec_tk.word_index)

# model
class enc_dec_lstm(object):
    def __init__(self):
        # self.checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH,
        #                                   save_best_only=True)
        # self.print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        self._build_model()
        self._load_weights(WEIGHT_FILE_PATH)

    def _build_model(self):
        # encoder
        self.enc_input = Input(shape=(INPUT_MAX_LEN,),
                               name='enc_input')
        self.enc_emb = Embedding(input_vocab, EMBEDDING_SIZE,
                                 weights=[input_embedding_matrix],
                                 trainable=False)
        self.enc_emb_res = self.enc_emb(self.enc_input)
        self.enc_lstm = LSTM(HIDDEN_UNITS,
                             return_state=True,
                             name='enc_lstm')
        self.enc_outputs, enc_state_h, enc_state_c = self.enc_lstm(self.enc_emb_res)
        self.enc_states = [enc_state_h, enc_state_c]

        # decoder
        self.dec_input = Input(shape=(None,),
                               name='dec_input')
        self.dec_emb = Embedding(target_vocab, EMBEDDING_SIZE,
                                 weights=[target_embedding_matrix],
                                 trainable=False)
        self.dec_emb_res = self.dec_emb(self.dec_input)
        self.dec_lstm = LSTM(HIDDEN_UNITS,
                             return_state=True,
                             return_sequences=True,
                             name='dec_lstm')
        self.lstm_outputs, _, _  = self.dec_lstm(self.dec_emb_res,
                                                 initial_state=self.enc_states)
        self.dec_dense = TimeDistributed(Dense(target_vocab,
                                               activation='softmax',
                                               name='dec_dense'))
        self.dec_outputs = self.dec_dense(self.lstm_outputs)

        # model
        self.model = Model([self.enc_input, self.dec_input],
                           self.dec_outputs)

    def _load_weights(self, WEIGHT_FILE_PATH):
        self.model.load_weights(WEIGHT_FILE_PATH)
        self.model.compile(loss=sparse_categorical_crossentropy,
                           optimizer=Adam(1e-3))

    def _preprocess(self, val_enc_sent):
        val_enc_sent = parse_q(val_enc_sent)
        x = enc_tk.texts_to_sequences([val_enc_sent])
        x = padding(x, INPUT_MAX_LEN)
        return x

    def guess(self, val_enc_sent):
        # encoder
        enc_model = Model(self.enc_input, self.enc_states)
        state_value = enc_model.predict(self._preprocess(val_enc_sent))

        # decoder
        dec_input = Input(shape=(1,))
        dec_state_input = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        dec_emb_res = self.dec_emb(dec_input)
        dec_lstm_outputs, state_h, state_c = self.dec_lstm(dec_emb_res,
                                                           initial_state=dec_state_input)
        dec_lstm_states = [state_h, state_c]
        dec_outputs = self.dec_dense(dec_lstm_outputs)
        dec_model = Model([dec_input]+dec_state_input,
                          [dec_outputs]+dec_lstm_states)

        # inference
        dec_sent = []
        target_curr = np.array([word2idx['start']])[None, :]

        stop_condition = False
        while not stop_condition:
            logits, hid_st_h, hid_st_c = dec_model.predict([target_curr]+state_value)
            wid = np.argmax(logits[0], axis=1)[0]
            word = idx2word[wid]
            if word !='start' and word !='end':
                dec_sent.append(word)

            target_curr = np.array([wid])[None, :]
            state_value = [hid_st_h, hid_st_c]

            if len(dec_sent)>TARGET_MAX_LEN or word=='end':
                stop_condition=True

        return ' '.join(dec_sent).strip()

    def test_run(self, nrounds = 5):
        for i in range(nrounds):
            input_sent = inputs[i]
            output_sent = self.guess(input_sent)
            print('input: ', input_sent)
            print('reply: ', output_sent)
            print('-------')


if __name__ == '__main__':
    print('load model...')
    model = enc_dec_lstm()
    print('test...')
    model.test_run()
    
    print('input: ', 'how are you?')
    output_sent = model.guess('how are you?')
    print('reply: ', output_sent)
    print('-------')
