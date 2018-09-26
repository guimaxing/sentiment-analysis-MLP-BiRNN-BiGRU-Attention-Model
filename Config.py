# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:29:25 2018
@author: Moc
"""

#定义参数
SENTENCE_NUM = 21105
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2


#训练
epochs = 10    
batch_size = 50 
#choice : BIGRU/BILSTM/MLP/ATENTION_LSTM/ATTENTION_GRU   
choice = 'BILSTM'


#预测
model_file = './MLP_model/MLP.h5'
#model_file = './lstm_model/lstm.h5'
#model_file = './ATENTION_LSTM_model/ATENTION_LSTM.h5'

string_list = ['跟想象中差太多，我自己买了100多的配件，你们太夸张了，太不满意了',
               '天气很好，非常开心',
               '在这家店买的东西质量很差，一点诚信都没有，不会再光顾了']

              