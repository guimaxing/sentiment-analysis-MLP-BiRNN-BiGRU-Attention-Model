# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:29:25 2018
@author: Moc
"""

import os
import jieba
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from Config import epochs,batch_size,choice
from Config import SENTENCE_NUM,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,EMBEDDING_DIM,VALIDATION_SPLIT
from Utils import model_select

#加载训练文件
def loadfile():
    neg=pd.read_excel('./data/mpk/neg.xls',header=None,index=None)
    pos=pd.read_excel('./data/mpk/pos.xls',header=None,index=None)

    combined=np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))

    return combined,y

#对句子经行分词，并去掉换行符
def split_sentence(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


#读取词向量，生成词典
def embedding_dict():
    embeddings_index = {}
    f = open('./data/zhwiki_2017_03.sg_50d.word2vec',encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index


#补齐数据维度
def data_pad(texts):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data,word_index


#构造数据
def data_classfier():
    combined,y = loadfile()
    texts = split_sentence(combined)
    labels = y
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', len(texts))
    print('Shape of label tensor:', len(labels))

    data,word_index = data_pad(texts)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return x_train,y_train,x_val,y_val,word_index

#模型选择
def model_(word_index,embeddings_index,choice):
    M = model_select()
    log_dir = './{}_model/log'.format(choice) 
    filepath = './{}_model/{}.h5'.format(choice,choice)
    if choice == 'BIGRU':
        model = M.BIGRU_model(word_index,embeddings_index)
    elif choice == 'BILSTM':
        model = M.BILSTM_model(word_index,embeddings_index)
    elif choice == 'MLP':
        model = M.MLP_model(word_index,embeddings_index)
    elif choice == 'ATENTION_LSTM':
        model = M.ATENTION_LSTM_model(word_index,embeddings_index)
    elif choice == 'ATTENTION_GRU':
        model = M.ATTENTION_GRU_model(word_index,embeddings_index)
    else:
        print('选择的模型未存在可选配置中，请选择BIGRU/BILSTM/MLP/ATENTION_LSTM/ATTENTION_GRU中的一个')
        #终止程序， os._exit(0) 正常退出
#        os._exit(0)
    return log_dir,filepath,model

def mdk(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

#构造模型，训练
def train(epochs,batch_size,choise):    
    x_train,y_train,x_val,y_val,word_index = data_classfier()
    embeddings_index = embedding_dict()
    log_dir,filepath,model = model_(word_index,embeddings_index,choise)
    print(log_dir, filepath)
    mdk(log_dir)

    print('Traing and validation set number of positive and negative reviews')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    tensorboard = TensorBoard(log_dir=log_dir)
    #保存最优模型
    checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',mode='max' ,save_best_only='True')

    callback_lists=[tensorboard,checkpoint]
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callback_lists)

    #测试集
    score = model.evaluate(x_val, y_val, batch_size=batch_size)
    print('loss: {}    acc: {}'.format(score[0], score[1]))


if __name__ == '__main__':
    epochs = epochs   
    batch_size = batch_size
    choice = choice
    train(epochs,batch_size,choice)
    
    