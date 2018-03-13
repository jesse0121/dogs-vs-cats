"""
用来训练模型，并输出对测试集的预测
"""
import pandas as pd
from keras.preprocessing.image import *
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
import h5py
import numpy as np




def load_vector(gap_name, seed=2017):
    """
    读取经过预训练保存的h5文件，h5文件中包含train, test, label
    :param gap_name: string or list 选取的imagenet模型,string应为电脑本地保存的h5文件名
    :param seed: int 随机数种子
    :return: ndarray 返回一个[samples, feature]的向量
    """
    np.random.seed(seed)

    x_train = []
    x_test = []

    for filename in gap_name:
        with h5py.File(filename, 'r') as h:
            x_train.append(np.array(h['train']))
            x_test.append(np.array(h['test']))

    with h5py.File('gap_ResNet50.h5', 'r') as h:
        y_train = np.array(h['label'])

    x_train = np.concatenate(x_train, axis=1)
    x_test = np.concatenate(x_test, axis=1)

    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train, x_test

def predict(x_test, model):
    """
    对测试集的数据进行预测，返回一个numpy array
    :param x_test: ndarray
    :param model: keras.Model 已经训练好的模型
    :return: numpy array
    """
    y_pred = model.predict(x_test, verbose=1)
    y_pred = y_pred.clip(min=0.005, max=0.995)
    return y_pred

def predict_to_csv(x_test, model):
    """
    对测试集的数据进行预测，返回一个DataFrame,并向当前路径输出一个与sampleSubmission格式相同的csv文件
    :param x_test: ndarray
    :param model: keras.Model 已经训练好的模型
    :return: pandas.DataFrame
    """
    y_pred = predict(x_test, model)
    df = pd.read_csv(r'row_data/sampleSubmission.csv')

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory('row_data/test', target_size=(224, 224), shuffle=False, follow_links=True,
                                             class_mode=None, batch_size=16)

    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('\\')+1 : fname.rfind('.')])
        df.loc[index-1, ['label']] = y_pred[i]
    df.to_csv('predict/pred.csv', index=False) #TODO:未来可以检测文件夹中是否有预测文件，若有在pred后+1，方便管理
    print(df.head())
    return df


if __name__ == '__main__':
    X_train, Y_train, X_test = load_vector(('gap_ResNet50.h5', 'gap_Xception.h5', 'gap_inceptionV3.h5'))
    np.random.seed(2017)
    input_tensor = Input(shape=X_train.shape[1:])
    x = Dropout(0.5)(input_tensor)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, x)

    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=9, validation_split=0.2)
    pred = predict_to_csv(X_test, model=model)