# coding:utf-8

import os
import string
from nltk.corpus import stopwords
from collections import Counter


class PrepareData(object):

    def __init__(self, pos_path, neg_path):
        self.__pos_path = pos_path
        self.__neg_path = neg_path

        self.__pos_list = []
        self.__neg_list = []

        self.__pos_label = None
        self.__neg_label = None

        self.__vob = None
        self.__tok = None

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

        def __update_vocab(list, vob):
            for words in list:
                vob.update(words)

            return vob

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

        self.__vob = __update_vocab(self.__pos_list, Counter())
        self.__vob = __update_vocab(self.__neg_list, self.__vob)

        print(self.__vob)


if __name__ == "__main__":
    pd = PrepareData(
        pos_path="D:\\Code\\Python\\NLP\\IMDB\\txt_sentoken\\pos",
        neg_path="D:\\Code\\Python\\NLP\\IMDB\\txt_sentoken\\neg"
    )
    pd.data_prepare()
