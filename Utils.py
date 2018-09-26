# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:29:25 2018
@author: Moc
"""
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model
from Attention_layer import Attention_layer

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from Config import SENTENCE_NUM,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,EMBEDDING_DIM,VALIDATION_SPLIT

class model_select():
	#embedding层	
    def emb_model_layer(self,word_index,embeddings_index):
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        print ('Length of embedding_matrix:', embedding_matrix.shape[0])
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    mask_zero=False,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)   
        return embedding_layer
    
    def BIGRU_model(self,word_index,embeddings_index):
        print('开始构建BIGRU模型')
        embedding_layer = self.emb_model_layer(word_index,embeddings_index)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        l_gru = Bidirectional(GRU(100, return_sequences=False))(embedded_sequences)
        dense_1 = Dense(100,activation='tanh')(l_gru)
        dense_2 = Dense(2, activation='softmax')(dense_1)
     
        model = Model(sequence_input, dense_2)
     
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
     
        model.summary()
        return model

    def BILSTM_model(self,word_index,embeddings_index):
        print('开始构建BILSTM模型')
        embedding_layer = self.emb_model_layer(word_index,embeddings_index)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        l_gru = Bidirectional(LSTM(100, return_sequences=False))(embedded_sequences)
        dense_1 = Dense(100,activation='tanh')(l_gru)
        dense_2 = Dense(2, activation='softmax')(dense_1)
     
        model = Model(sequence_input, dense_2)
     
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
     
        model.summary()
        return model

    def MLP_model(self,word_index,embeddings_index):
        print('开始构建MLP模型')
        embedding_layer = self.emb_model_layer(word_index,embeddings_index)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        dense_1 = Dense(100,activation='tanh')(embedded_sequences)
        max_pooling = GlobalMaxPooling1D()(dense_1)
        dense_2 = Dense(2, activation='softmax')(max_pooling)
     
        model = Model(sequence_input, dense_2)
     
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
     
        model.summary()
        return model
        
        
    def ATENTION_LSTM_model(self,word_index,embeddings_index):
        print('开始构建MLP模型')
        embedding_layer = self.emb_model_layer(word_index,embeddings_index)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
        l_att = Attention_layer()(l_lstm)
        dense_1 = Dense(100,activation='tanh')(l_att)
        dense_2 = Dense(2, activation='softmax')(dense_1)
        model = Model(sequence_input, dense_2)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        model.summary()
        return model

    def ATENTION_GRU_model(self,word_index,embeddings_index):
        print('开始构建MLP模型')
        embedding_layer = self.emb_model_layer(word_index,embeddings_index)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
        l_att = Attention_layer()(l_gru)
        dense_1 = Dense(100,activation='tanh')(l_att)
        dense_2 = Dense(2, activation='softmax')(dense_1)
        model = Model(sequence_input, dense_2)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        model.summary()
        return model