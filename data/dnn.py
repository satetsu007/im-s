# coding:utf-8

from model import VGG16_model
from util import load_data
from keras.callbacks import ModelCheckpoint
import os
from keras.models import model_from_json
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.backend import tensorflow_backend as backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
backend.set_session(session)

def train(ModelStr=""):
    
    hist = []
    
    for ho in range(2):
        
        model = VGG16_model()
        save_model(model, ho, ModelStr)
        
        x_train, y_train, x_valid, y_valid = load_data("train")
        
        cp = ModelCheckpoint('./cache/' + ModelStr + '/model_weights_%s_%i_{epoch:02d}.h5'%(ModelStr, ho),
                             monitor='val_loss', save_best_only=False)
        
        hist.append(model.fit(x_train, y_train, batch_size=32,
                              epochs=100,
                              verbose=1,
                              validation_data=(x_valid, y_valid),
                              shuffle=True,
                              callbacks=[cp]))

def save_model(model, ho, ModelStr=""):
    # モデルオブジェクトをjson形式に変換
    json_string = model.to_json()
    # カレントディレクトリにcacheディレクトリがなければ作成
    if not os.path.isdir('cache/' + ModelStr):
        os.mkdir('cache/' + ModelStr)
    # モデルの構成を保存するためのファイル名
    json_name = 'architecture_%s_%i.json'%(ModelStr, ho)
    # モデル構成を保存
    open(os.path.join('cache', json_name), 'w').write(json_string)
  
# モデルの構成と重みを読み込む

def read_model(ho, ModelStr='', epoch='00'):
    
    # モデル構成のファイル名
    json_name = 'architecture_%s_%i.json'%(ModelStr, ho)
    # モデル重みのファイル名
    weight_name = ModelStr + '/model_weights_%s_%i_%s.h5'%(ModelStr, ho, epoch)
    # モデルの構成を読込み、jsonからモデルオブジェクトへ変換
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    # モデルオブジェクトへ重みを読み込む
    model.load_weights(os.path.join('cache', weight_name))
    
    return model


def test(ModelStr, epoch1, epoch2):
    
    x_test, y_test = load_data("test")
    tmp = 0
    class_name = os.listdir("test")
    
    for test_class in range(y_test.shape[1]):
        
        
        backend.clear_session()
        
        class_data = []
        file_name = os.listdir("test/" + class_name[test_class])
        yfull_test = []
        
        for i in range(y_test.shape[0]):
            
            if y_test[i, test_class] == True:
                class_data.append(x_test[i])
        
        class_data = np.array(class_data)
        
        for ho in range(2):
            
            if ho == 0:
                epoch_n = epoch1
            else:
                epoch_n = epoch2
            
            model = read_model(ho, ModelStr, epoch_n)
            
            score = model.predict_proba(class_data, batch_size=128, verbose=1)
            
            yfull_test.append(score)
        
        test_res = np.array(yfull_test[0])
        test_res += np.array(yfull_test[1])
        test_res /= 2
        
        result = pd.DataFrame(test_res, columns=class_name)
        result.loc[:, "img"] = pd.Series(file_name, index=result.index)
        
        if not os.path.isdir("subm/" + ModelStr):
            os.mkdir("subm/" + ModelStr)
        if not os.path.isdir("subm/" + ModelStr + "/" + str(int(epoch1) + 1) + "," + str(int(epoch2) + 1)):
            os.mkdir("subm/" + ModelStr + "/" + str(int(epoch1) + 1) + "," + str(int(epoch2) + 1))
        sub_file = "./subm/" + ModelStr + "/" + str(int(epoch1) + 1) + "," + str(int(epoch2) + 1) + "/result_%s_%s.csv" %(ModelStr, class_name[test_class])
        
        # print(result.shape)
        # print(result.head)
        result = result.ix[:, ["img"] + [i for i in class_name]]
        result.to_csv(sub_file, index = False)
        
        one_column = np.where(np.argmax(test_res, axis=1)==test_class)
        
        tmp += len(one_column[0])
        
        print("\n---------------------------------")
        print("\nclass:" + class_name[test_class])
        print("\n正解数　　　" + str(len(one_column[0])))
        print("不正解数　　" + str(test_res.shape[0] - len(one_column[0])))
        print("正解率　　　" + str(len(one_column[0]) / test_res.shape[0] * 100 ) + "%\n")
        print("---------------------------------")
    
    print("\n総正解数　　" + str(tmp))
    print("総不正解数　" + str(x_test.shape[0] - tmp ))
    print("総正解率　　" + str(tmp / x_test.shape[0] * 100) + "%\n")
    print("---------------------------------")




# 実行した際に呼ばれる
if __name__ == '__main__':
    # 引数を取得
    # [1] = train or test
    # [2] = test時のみ、使用Epoch数 1
    # [3] = test時のみ、使用Epoch数 2
    param = sys.argv
    if len(param) < 2:
        sys.exit ("Usage: python dnn.py [ModelStr] [train, test] [1] [2]")
    # train or test
    ModelStr = param[1]
    run_type = param[2]
    if run_type == 'train':
        train(ModelStr)
    elif run_type == 'test':
        # testの場合、使用するエポック数を引数から取得する
        if len(param) == 5:
            epoch1 = "%02d"%(int(param[3])-1)
            epoch2 = "%02d"%(int(param[4])-1)
            test(ModelStr, epoch1, epoch2)
        else:
            sys.exit ("Usage: dnn.py [ModelStr] [train, test] [1] [2]")
    else:
        sys.exit ("Usage: python dnn.py [ModelStr] [train, test] [1] [2]")


