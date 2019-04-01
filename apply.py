#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from SpeechModel251 import ModelSpeech
from LanguageModel import ModelLanguage
import pandas as pd
import os
import Levenshtein

def pinyin_decode(pinyin):
    id_to_tok = {"yi":"1", "er":"2", "san":"3" ,"si":"4", "wu":"5","liu":"6","qi":"7","ba":"8", "jiu":"9","ling":"0"}
    tok = [id_to_tok[p[:-1]] for p in pinyin ] # 忽略音调
    
    return ''.join(tok)

def recognize_same(x, content, ms):
    
    # 识别的随机码与真实的随机码的距离是否在1以内
    # 距离在1以内的则认为验证码识别通过
    
    r = ms.RecognizeSpeech_FromFile(x)
    r = pinyin_decode(r)
    dist = Levenshtein.distance(content,r)
    return dist <= 1

    
def recognize_code(x, content, ms):
    
    # 返回识别的随机码
    
    r = ms.RecognizeSpeech_FromFile(x)
    r = pinyin_decode(r)
    
    return r
    

if __name__=='__main__':
    
    datapath = 'model_speech'
    modelpath = 'model_speech'
    ms = ModelSpeech(datapath)

    ms.LoadModel(os.path.join(modelpath, 'speech_model251_e_0_step_200.model'))

    filename = '/data1/1806_speech-recognition/random_code/wav/00818_VID_20190328_145629.wav'

    content = '00818'
    
    print(recognize_same(filename, content, ms))
    print(recognize_code(filename, content, ms))

















