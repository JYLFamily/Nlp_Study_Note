# coding:utf-8

import os
import string
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
np.random.seed(7)


class PrepareDataAndPreTrainModel(object):

    def __init__(self, *, pos_path, neg_path, glove_path):
        # init
        self.__pos_path = pos_path
        self.__neg_path = neg_path
        self.__glove_path = glove_path

        # data prepare
        self.__pos_list = []
        self.__neg_list = []

        self.__pos_label = None
        self.__neg_label = None

        self.__train, self.__train_label = [None for _ in range(2)]
        self.__test, self.__test_label = [None for _ in range(2)]

        self.__tok = None

        # glove prepare
        self.__embeddings_index = None  # word: glove vector

        # model fit
        self.__embedding_matrix = None
        self.__net = None

    def data_prepare(self):
        def __load_text(path):
            """
            :param path: .txt 文件路径
            :return: text str
            """
            file = open(path, "r")
            text = file.read()
            file.close()

            return text

        def __clean_text(text):
            """
            :param text: text str
            :return: words list
            """
            # 通过空格分词
            tokens = text.split()
            # 转换成小写
            tokens = [w.lower() for w in tokens]
            # 去掉 word 中的标点符号
            stripped = [w.translate(str.maketrans("", "", string.punctuation)) for w in tokens]
            # 去掉非字母组成的 word
            words = [word for word in stripped if word.isalpha()]
            # 去掉停用词
            words = [word for word in words if word not in set(stopwords.words("english"))]
            # 去掉单个字母的 word
            words = [word for word in words if len(word) > 1]

            return words

        for filename in os.listdir(self.__pos_path):
            text = __load_text(os.path.join(self.__pos_path, filename))
            words = __clean_text(text)

            self.__pos_list.append(words)

        for filename in os.listdir(self.__neg_path):
            text = __load_text(os.path.join(self.__neg_path, filename))
            words = __clean_text(text)

            self.__neg_list.append(words)

        self.__pos_label = [1 for _ in range(len(self.__pos_list))]
        self.__neg_label = [1 for _ in range(len(self.__neg_list))]

        self.__train = self.__pos_list[0:900] + self.__neg_list[0:900]
        self.__train_label = self.__pos_label[0:900] + self.__neg_label[0:900]
        self.__test = self.__pos_list[900:] + self.__neg_list[900:]
        self.__test_label = self.__pos_label[900:] + self.__neg_label[900:]

        self.__tok = Tokenizer()
        self.__tok.fit_on_texts(self.__train)
        self.__train = self.__tok.texts_to_sequences(self.__train)
        self.__test = self.__tok.texts_to_sequences(self.__test)

        self.__train = pad_sequences(self.__train, maxlen=500)  # maxlen 序列长度
        self.__test = pad_sequences(self.__test, maxlen=500)

    def glove_prepare(self):
        self.__embeddings_index = dict()
        with open(self.__glove_path, mode="r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                values = line.split()
                word = values[0]
                self.__embeddings_index[word] = np.array(values[1:], dtype="float32")
                line = f.readline()

    def model_fit(self):
        self.__embedding_matrix = np.zeros((len(self.__tok.word_index)+1, 50))
        for word, i in self.__tok.word_index.items():
            embedding_vector = self.__embeddings_index.get(word)
            if embedding_vector is not None:
                self.__embedding_matrix[i] = embedding_vector

        self.__net = Sequential()
        self.__net.add(Embedding(
            input_dim=len(self.__tok.word_index)+1,
            output_dim=50,
            weights=[self.__embedding_matrix],
            input_length=500,
            trainable=True
        ))
        self.__net.add(LSTM(100))
        self.__net.add(Dense(1, activation="sigmoid"))
        self.__net.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.__net.fit(
            self.__train,
            self.__train_label,
            epochs=3,
            batch_size=64
        )

    def model_predict(self):
        print(self.__net.evaluate(
            self.__test,
            self.__test_label,
            verbose=0
        ))


if __name__ == "__main__":
    pdatm = PrepareDataAndPreTrainModel(
        pos_path="D:\\Code\\Python\\NLP\\IMDB\\txt_sentoken\\pos",
        neg_path="D:\\Code\\Python\\NLP\\IMDB\\txt_sentoken\\neg",
        glove_path="D:\\Code\\Python\\NLP\\IMDB\\glove\\glove.6B.50d.txt"
    )
    pdatm.data_prepare()
    pdatm.glove_prepare()
    pdatm.model_fit()
    pdatm.model_predict()