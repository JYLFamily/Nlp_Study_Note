# coding:utf-8

import gc
import os
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.losses import msle
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
np.random.seed(7)


def rmsle(y_true, y_pred):
    import keras.backend as K
    return K.sqrt(msle(y_true, y_pred))


class PrepareOverviewRnn(object):

    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label, self.__test_label = [None for _ in range(2)]

        self.__tok = None
        self.__mle = None
        self.__folds = None
        self.__oof_preds = None
        self.__sub_preds = None

    def read_data(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"))

    def prepare_data(self):
        self.__train_feature = self.__train[["overview"]].copy()
        self.__test_feature = self.__test[["overview"]].copy()
        self.__train_label = self.__train["revenue"].copy()
        del self.__train, self.__test
        gc.collect()

        # train clean
        # 去除非字母符号
        self.__train_feature["overview"] = \
            self.__train_feature["overview"].str.lower().str.replace(r"[^a-zA-Z]", " ")
        # 分词
        self.__train_feature["overview"][~self.__train_feature["overview"].isna()] = \
            self.__train_feature["overview"][~self.__train_feature["overview"].isna()].apply(
                lambda text: word_tokenize(text))
        # 去除非单词
        self.__train_feature["overview"][~self.__train_feature["overview"].isna()] = \
            self.__train_feature["overview"][~self.__train_feature["overview"].isna()].apply(
                lambda words: [word for word in words if word.isalpha()])
        # 去除停用词
        self.__train_feature["overview"][~self.__train_feature["overview"].isna()] = \
            self.__train_feature["overview"][~self.__train_feature["overview"].isna()].apply(
                lambda words: [word for word in words if word not in stopwords.words("english")])
        # 词形还原 词干提取
        self.__train_feature["overview"][~self.__train_feature["overview"].isna()] = \
            self.__train_feature["overview"][~self.__train_feature["overview"].isna()].apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="n") for word in words]).apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="v") for word in words]).apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="a") for word in words])
        # 拼接 list 成 string 以便 vectorizer 使用
        self.__train_feature["overview"][~self.__train_feature["overview"].isna()] = \
            self.__train_feature["overview"][~self.__train_feature["overview"].isna()].apply(
                lambda words: " ".join(words))
        # 填充缺失值
        self.__train_feature["overview"] = self.__train_feature["overview"].fillna(" ")

        # test clean
        self.__test_feature["overview"] = \
            self.__test_feature["overview"].str.lower().str.replace(r"[^a-zA-Z]", " ")

        self.__test_feature["overview"][~self.__test_feature["overview"].isna()] = \
            self.__test_feature["overview"][~self.__test_feature["overview"].isna()].apply(
                lambda text: word_tokenize(text))

        self.__test_feature["overview"][~self.__test_feature["overview"].isna()] = \
            self.__test_feature["overview"][~self.__test_feature["overview"].isna()].apply(
                lambda words: [word for word in words if word.isalpha()])

        self.__test_feature["overview"][~self.__test_feature["overview"].isna()] = \
            self.__test_feature["overview"][~self.__test_feature["overview"].isna()].apply(
                lambda words: [word for word in words if word not in stopwords.words("english")])

        self.__test_feature["overview"][~self.__test_feature["overview"].isna()] = \
            self.__test_feature["overview"][~self.__test_feature["overview"].isna()].apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="n") for word in words]).apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="v") for word in words]).apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="a") for word in words])

        self.__test_feature["overview"][~self.__test_feature["overview"].isna()] = \
            self.__test_feature["overview"][~self.__test_feature["overview"].isna()].apply(
                lambda words: " ".join(words))

        self.__test_feature["overview"] = self.__test_feature["overview"].fillna(" ")

        self.__tok = Tokenizer(split=" ")
        self.__tok.fit_on_texts(self.__train_feature["overview"])
        self.__train_feature = self.__tok.texts_to_sequences(self.__train_feature["overview"])
        self.__test_feature = self.__tok.texts_to_sequences(self.__test_feature["overview"])

        self.__mle = int(np.percentile([len(element) for element in self.__train_feature], [95]))
        self.__train_feature = pad_sequences(self.__train_feature, maxlen=self.__mle)
        self.__test_feature = pad_sequences(self.__test_feature, maxlen=self.__mle)

        self.__train_label = self.__train_label.values

    def fit_predict_model(self):
        embeddings_index = dict()
        with open("E:\\Kaggle\\TMDB_Box_Office_Prediction\\glove\\glove.6B.50d.txt", mode="r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                values = line.split()
                word = values[0]
                embeddings_index[word] = np.array(values[1:], dtype="float32")
                line = f.readline()

        embedding_matrix = np.zeros((len(self.__tok.word_index) + 1, 50))
        for word, i in self.__tok.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.__folds = KFold(n_splits=5, shuffle=True, random_state=7)
        self.__oof_preds = np.zeros(shape=self.__train_feature.shape[0])
        self.__sub_preds = np.zeros(shape=self.__test_feature.shape[0])

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(self.__train_feature, self.__train_label)):
            trn_x, trn_y = self.__train_feature[trn_idx], self.__train_label[trn_idx]
            val_x, val_y = self.__train_feature[val_idx], self.__train_label[val_idx]

            net = Sequential()
            net.add(Embedding(
                input_dim=len(self.__tok.word_index) + 1,
                output_dim=50,
                weights=[embedding_matrix],
                input_length=self.__mle,
                trainable=False
            ))
            net.add(SimpleRNN(units=2))  # overfitting
            net.add(Dense(1, activation="linear"))
            net.compile(loss=rmsle, optimizer=Adam())
            net.fit(
                x=trn_x,
                y=np.log1p(trn_y),
                batch_size=32,
                epochs=10,
                verbose=2,
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)],
                validation_data=(val_x, val_y)
            )

            pred_val = np.expm1(net.predict(val_x)).reshape(-1, )  # predict shape (, 1)
            pred_test = np.expm1(net.predict(self.__test_feature)).reshape(-1, )

            self.__oof_preds[val_idx] = pred_val
            self.__sub_preds += pred_test / self.__folds.n_splits

            del trn_x, trn_y, val_x, val_y
            gc.collect()

    def write_data(self):
        pd.Series(self.__oof_preds) \
            .to_frame("train_rnn_overview") \
            .to_csv(os.path.join(self.__output_path, "train_rnn_overview.csv"), index=False)
        pd.Series(self.__sub_preds) \
            .to_frame("test_rnn_overview") \
            .to_csv(os.path.join(self.__output_path, "test_rnn_overview.csv"), index=False)


if __name__ == "__main__":
    por = PrepareOverviewRnn(
        input_path="E:\\Kaggle\\TMDB_Box_Office_Prediction\\raw",
        output_path="E:\\Kaggle\\TMDB_Box_Office_Prediction\\output"
    )
    por.read_data()
    por.prepare_data()
    por.fit_predict_model()
    por.write_data()