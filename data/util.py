# coding:utf-8

import numpy as np
from keras.utils.np_utils import to_categorical
import cv2
import os
import random


def load_data(kind):
    if kind == "train":
        
        train_data, train_label, validation_data, validation_label = read_data("train")
        
        train_label = convert_label("train", train_label)
        validation_label = convert_label("train", validation_label)
        
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        validation_data = np.array(validation_data)
        validation_label = np.array(validation_label)
        
        train_label = to_categorical(train_label)
        validation_label = to_categorical(validation_label)
        
        x_train = train_data
        x_valid = validation_data
        y_train = train_label
        y_valid = validation_label
        
        return x_train, y_train, x_valid, y_valid
    
    
    elif kind == "test":
        test_data, test_label = read_data("test")
        
        test_label = convert_label("test", test_label)
        
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        
        test_label = to_categorical(test_label)
        
        x_test = test_data
        y_test = test_label
        
        return x_test, y_test

# 画像ファイルの読み込み
def get_img(file):
    img = cv2.imread(file)  
    return img

# データセットの読み込み
def read_data(kind):
    
    if kind == "train":
        train_data = []
        train_label = []
        validation_data = []
        validation_label = []
        category = os.listdir(kind)
        
        for folder in category:
            file_list = os.listdir("%s/%s" % (kind, folder))
            
            random.shuffle(file_list)
            a = len(file_list) // 2
            if not 2 * a == len(file_list):
                b = a + 1
            for file in file_list[:a]:
                img = get_img("%s/%s/" % (kind, folder) + file)
                # height = img.shape[0]
                # width = img.shape[1]
                # img = cv2.resize(img,(48, 48))
                # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
                # img = img.reshape(1, img.shape[0] * img.shape[1] * img.shape[2]).astype("float32")[0]
                # 出来上がった配列をtrain_dataに追加。
                train_data.append(img / 255.)
                train_label.append(folder)
    
            for file in file_list[b:]:
                img = get_img("%s/%s/" % (kind, folder) + file)
                # height = img.shape[0]
                # width = img.shape[1]
                # img = cv2.resize(img,(48, 48))
                # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
                # img = img.reshape(1, img.shape[0] * img.shape[1] * img.shape[2]).astype("float32")[0]
                # 出来上がった配列をtrain_dataに追加。
                validation_data.append(img / 255.)
                validation_label.append(folder)
            
        return train_data, train_label, validation_data, validation_label
        
    elif kind == "test":
        test_data = []
        test_label = []
        category = os.listdir(kind)
        
        for folder in category:
            file_list = os.listdir("%s/%s" % (kind, folder))
            
            for file in file_list:
                img = get_img("%s/%s/" % (kind, folder) + file)
                # img = cv2.resize(img,(48,48))
                # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
                # img = img.reshape(1, img.shape[0] * img.shape[1] * img.shape[2]).astype("float32")[0]
                # 出来上がった配列をtest_dataに追加。
                test_data.append(img / 255.)
                test_label.append(folder)
    
        return test_data, test_label
    
def convert_label(kind, label):
    category = os.listdir(kind)
    
    for i,l in enumerate(label):
        for j,c in enumerate(category):
            if l == c:
                label[i] = j
    
    return label
