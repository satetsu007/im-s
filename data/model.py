# cofing:utf-8

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD, RMSprop

def VGG16_model():
    # CNNを使用した多クラス分類
    # VGG-16を模倣
    model = Sequential()
    # 入力: サイズが100x100で3チャンネルをもつ画像 -> (200, 200, 3) のテンソル
    # それぞれのlayerで3x3の畳み込み処理を適用している
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last", input_shape=(200, 200, 3)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation("tanh"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    
    return model
