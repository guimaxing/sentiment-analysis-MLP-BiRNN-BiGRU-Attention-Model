# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:29:25 2018
@author: Moc
"""

import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from Attention_layer import Attention_layer
from Config import SENTENCE_NUM,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,EMBEDDING_DIM,VALIDATION_SPLIT
from Config import model_file,string_list

#对句子经行分词，并去掉换行符
def split_sentence(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

#预测
def predict_result(model, string):
    tx = [string]
    txs = split_sentence(tx)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(txs)
    sequences = tokenizer.texts_to_sequences(txs)
#    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    result = model.predict(data)
    result_0 = result[0][0]
    result_1 = result[0][1]
    return result_0, result_1
    
if __name__ == '__main__':
    """
    添加自定义损失或者网络层
    tips:
    load_model函数提供了custom_objects参数，所以加载时需要加入这个参数
    
    假设自定义参数loss的函数名为cosloss,所以加载时应采用以下方式
    from * import cosloss
    model = load_model(model_file, {'cosloss':cosloss})
    
    假设自定义网络层的函数名为Attention_layer,所以加载时应采用以下方式
    from Attention_layer import Attention_layer
    model = load_model(model_file,{'Attention_layer':Attention_layer})
    """
        
    model_file = model_file
    string_list = string_list
    
    print ('loading model......')
    model = load_model(model_file,{'Attention_layer':Attention_layer})
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',metrics=['acc'])
    # model.summary()
    print('--------------------------------')
    print('预测结果')

    for string in string_list:
        result_0, result_1 = predict_result(model, string)
        if result_0 > result_1:
        	print('第{}段文字预测为0的概率为{}'.format(string_list.index(string)+1,result_0))
        else:
        	print('第{}段文字预测为1的概率为{}'.format(string_list.index(string)+1,result_1))
    