# coding:utf-8

import gc
import os
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import make_scorer
np.random.seed(7)


def rmsle(y_true, y_pred):

    return -np.sqrt(np.average((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


class PrepareTagLine(object):

    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label, self.__test_label = [None for _ in range(2)]

        self.__folds = None
        self.__oof_preds = None
        self.__sub_preds = None

    def read_data(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"))

    def prepare_data(self):
        self.__train_feature = self.__train[["tagline"]].copy()
        self.__test_feature = self.__test[["tagline"]].copy()
        self.__train_label = self.__train["revenue"].copy()
        del self.__train, self.__test
        gc.collect()

        # train clean
        # 去除非字母符号
        self.__train_feature["tagline"] = \
            self.__train_feature["tagline"].str.lower().str.replace(r"[^a-zA-Z]", " ")
        # 分词
        self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()] = \
            self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()].apply(
                lambda text: word_tokenize(text))
        # 去除非单词
        self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()] = \
            self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()].apply(
                lambda words: [word for word in words if word.isalpha()])
        # 去除停用词
        self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()] = \
            self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()].apply(
                lambda words: [word for word in words if word not in stopwords.words("english")])
        # 词形还原 词干提取
        self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()] = \
            self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()].apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="n") for word in words]).apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="v") for word in words]).apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="a") for word in words]).apply(
                lambda words: [PorterStemmer().stem(word) for word in words])
        # 拼接 list 成 string 以便 vectorizer 使用
        self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()] = \
            self.__train_feature["tagline"][~self.__train_feature["tagline"].isna()].apply(
                lambda words: " ".join(words))
        # 填充缺失值
        self.__train_feature["tagline"] = self.__train_feature["tagline"].fillna(" ")

        # test clean
        self.__test_feature["tagline"] = \
            self.__test_feature["tagline"].str.lower().str.replace(r"[^a-zA-Z]", " ")

        self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()] = \
            self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()].apply(
                lambda text: word_tokenize(text))

        self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()] = \
            self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()].apply(
                lambda words: [word for word in words if word.isalpha()])

        self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()] = \
            self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()].apply(
                lambda words: [word for word in words if word not in stopwords.words("english")])

        self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()] = \
            self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()].apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="n") for word in words]).apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="v") for word in words]).apply(
                lambda words: [WordNetLemmatizer().lemmatize(word, pos="a") for word in words]).apply(
                lambda words: [PorterStemmer().stem(word) for word in words])

        self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()] = \
            self.__test_feature["tagline"][~self.__test_feature["tagline"].isna()].apply(
                lambda words: " ".join(words))

        self.__test_feature["tagline"] = self.__test_feature["tagline"].fillna(" ")

        # count vectorizer
        # from sklearn.feature_extraction.text import CountVectorizer
        # vectorizer = CountVectorizer(lowercase=False)
        # vectorizer.fit(self.__train_feature["tagline"])
        # self.__train_feature = vectorizer.transform(self.__train_feature["tagline"])
        # self.__test_feature = vectorizer.transform(self.__test_feature["tagline"])

        # tfidf vectorizer bad
        vectorizer = TfidfVectorizer(lowercase=False)
        vectorizer.fit(self.__train_feature["tagline"])
        self.__train_feature = vectorizer.transform(self.__train_feature["tagline"])
        self.__test_feature = vectorizer.transform(self.__test_feature["tagline"])

        # min max scale bad
        scaler = MinMaxScaler()
        scaler.fit(self.__train_feature.toarray())
        self.__train_feature = scaler.transform(self.__train_feature.toarray())
        self.__test_feature = scaler.transform(self.__test_feature.toarray())
        self.__train_label = self.__train_label.values

    def fit_predict_model(self):
        # lm = LinearRegression()
        # lm.fit(self.__train_feature, np.log1p(self.__train_label.values))
        # print(
        #     np.sqrt(mean_squared_error(self.__train_label.values, np.expm1(lm.predict(self.__train_feature)))))

        self.__folds = KFold(n_splits=5, shuffle=True, random_state=7)
        self.__oof_preds = np.zeros(shape=self.__train_feature.shape[0])
        self.__sub_preds = np.zeros(shape=self.__test_feature.shape[0])

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(self.__train_feature, self.__train_label)):
            idx = np.zeros(shape=(self.__train_feature.shape[0], ))
            idx[trn_idx] = -1

            ps = PredefinedSplit(idx)
            clf = GridSearchCV(
                estimator=HuberRegressor(),
                param_grid={"epsilon": [1.25, 1.35, 1.45], "alpha": [0.00001, 0.0001, 0.001]},
                scoring=make_scorer(rmsle),
                n_jobs=2,
                refit=False,
                cv=ps)
            clf.fit(self.__train_feature, np.log1p(self.__train_label))

            clf = HuberRegressor(** clf.best_params_)
            clf.fit(self.__train_feature[trn_idx], np.log1p(self.__train_label[trn_idx]))
            pred_val = np.expm1(clf.predict(self.__train_feature[val_idx]))
            pred_test = np.expm1(clf.predict(self.__test_feature))

            self.__oof_preds[val_idx] = pred_val
            self.__sub_preds += pred_test / self.__folds.n_splits

            print("Fold %2d RMSE : %.2f" %
                  (n_fold + 1, rmsle(self.__train_label[val_idx], self.__oof_preds[val_idx])))

    def write_data(self):
        pd.Series(self.__sub_preds)  \
            .to_frame("tagline_bow") \
            .to_csv(os.path.join(self.__output_path, "tagline_bow.csv"), index=False)


if __name__ == "__main__":
    ptl = PrepareTagLine(
        input_path="E:\\Kaggle\\TMDB_Box_Office_Prediction\\raw",
        output_path="E:\\Kaggle\\TMDB_Box_Office_Prediction\\output"
    )
    ptl.read_data()
    ptl.prepare_data()
    ptl.fit_predict_model()
    ptl.write_data()