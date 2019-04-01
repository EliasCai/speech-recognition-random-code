#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于测试整个一套语音识别系统的程序
语音模型 + 语言模型
"""
# import platform as plat

from SpeechModel251 import ModelSpeech
from LanguageModel import ModelLanguage
import pandas as pd
import os
import Levenshtein

datapath = 'model_speech'
modelpath = 'model_speech'

def pinyin_decode(pinyin):
    id_to_tok = {"yi":"1", "er":"2", "san":"3" ,"si":"4", "wu":"5","liu":"6","qi":"7","ba":"8", "jiu":"9","ling":"0"}
    tok = [id_to_tok[p[:-1]] for p in pinyin ] # 忽略音调
    
    return ''.join(tok)
    
ms = ModelSpeech(datapath)

ms.LoadModel(os.path.join(modelpath, 'speech_model251_e_0_step_200.model'))

r = ms.RecognizeSpeech_FromFile('/data1/1806_speech-recognition/random_code/wav/09526_VID_20190329_112355.wav')

print('*[提示] 语音识别结果：\n',r)

print(pinyin_decode(r))


def recog(x):
    
    # code = os.path.basename(x)[:5]s
    r = ms.RecognizeSpeech_FromFile(x)
    r = pinyin_decode(r)
    return r

# r = ml.SpeechToText(str_pinyin)

test_files = pd.read_csv('test_wav.csv')
test_files['content'] = test_files['content'].astype('str')        
test_files['content'] = test_files['path'].map(lambda x: os.path.basename(x)[:5])


test_files['predict'] = test_files['path'].map(recog)

result = test_files['predict'] == test_files['content']

print(result.mean())

test_files.to_csv('pred.csv', index=False)

test_files['dist'] = test_files.apply(lambda x: Levenshtein.distance(x['content'],x['predict']), axis=1)














