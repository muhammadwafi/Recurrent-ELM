#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# src » preprocessing.py
# ==============================================
# @Author    : Muhammad Wafi <mwafi@mwprolabs.com>
# @Support   : [https://mwprolabs.com]
# @Created   : 04-07-2022
# @Modified  : 04-07-2022 12:47:10 pm
# ----------------------------------------------
# @Copyright (c) 2022 MWprolabs https://mwprolabs.com
#
###

import os
import re
import email
import glob
import collections
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tfidf import TfIdf
import config


class Preprocessing:
    """Sets of preprocessing functions"""

    def __init__(self, phishing_path, non_phishing_path):
        self.phishing_path = phishing_path
        self.non_phishing_path = non_phishing_path
        self.tfidf_vect = TfidfVectorizer(
            min_df=1,
            sublinear_tf=False,
            ngram_range=(1, 1),
            smooth_idf=False,
            norm=None
        )
        self.porter_stemmer = PorterStemmer()
        self.clean_regex = re.compile("[^a-zA-Z]")

    def case_folding(self, item):
        return item.lower()

    def cleaning(self, item):
        return re.sub(self.clean_regex, " ", item)

    def stopwords_and_stemming(self, item):
        data_arr = item.split()
        result = list()
        for word in data_arr:
            temp = word
            if len(temp) == 2:
                temp = self.remove_duplicate(word)
            if temp not in stopwords.words("english") and len(temp) > 1:
                result.append(self.porter_stemmer.stem(temp))
        return result

    def remove_duplicate(self, str):
        return "".join(set(str))

    def remove_html_tag(self, item):
        return BeautifulSoup(item, "lxml").text

    def get_file_names(self, path, filetype=None):
        if filetype:
            path += f"/*.{filetype}"
        else:
            path += "/*"

        file_list = glob.glob(path)
        # print(file_list)
        return file_list[:10]

    def open_email(self, paths):
        emails = list()
        for path in paths:
            print("opening", path)
            message = email.message_from_file(open(path, encoding="cp1252"))
            payload = message.get_payload()
            body = self.get_body_email(payload)
            emails.append(body)
        return emails

    def get_body_email(self, data):
        if type(data) == str:
            return data
        return self.get_body_email(data[0].get_payload())

    def get_fitur(self, datas):
        fitur = []
        for data in datas:
            fitur += data
        return list(set(fitur))

    def _get_cleaned_data(self, data_path, data_type):
        file_list = self.get_file_names(data_path, None)
        if data_type == "phising":
            file_list = self.get_file_names(data_path, "eml")

        mail_dt = self.open_email(file_list)
        cleaned_dt = []

        for mail_body in mail_dt:
            cleaned_html = self.remove_html_tag(mail_body)
            case_folded = self.case_folding(cleaned_html)
            cleaned = self.cleaning(case_folded)
            stopword_removed_and_stemmed = self.stopwords_and_stemming(cleaned)
            cleaned_dt.append(stopword_removed_and_stemmed)

        return cleaned_dt

    def get_tf_idf(self, dataset):
        fitur = self.get_fitur(dataset)
        tfidf_obj = TfIdf(dataset, fitur).getTfIdf()
        # ----- NEW ------
        # word_vectorizer = self.tfidf_vect
        # word_vectorizer.fit(fitur)
        # dt = word_vectorizer.transform(fitur)

        return tfidf_obj

    def get_pca(self, dataset):
        tfidf = self.get_tf_idf(dataset)
        pca_obj = PCA(n_components=10, random_state=42)
        # pca_obj = TruncatedSVD(n_components=10, random_state=42)
        pca_obj.fit(tfidf)

        return pca_obj.transform(tfidf)

    def get_phishing_df(self):
        cleaned_phising_dt = self._get_cleaned_data(
            self.phishing_path, "phising")
        p_pca_dt = self.get_pca(cleaned_phising_dt)
        cols = [f"X{i+1}" for i in range(len(p_pca_dt[0]))]

        # Create dataframe
        p_df = pd.DataFrame(data=p_pca_dt, columns=cols)
        # append labels
        p_df["labels"] = [1 for _ in range(len(p_pca_dt))]

        return p_df

    def get_non_phishing_df(self):
        cleaned_non_phising_dt = self._get_cleaned_data(
            self.non_phishing_path, "non_phishing")
        n_pca_dt = self.get_pca(cleaned_non_phising_dt)
        cols = [f"X{i+1}" for i in range(len(n_pca_dt[0]))]

        # Create dataframe
        n_df = pd.DataFrame(data=n_pca_dt, columns=cols)
        # append labels
        n_df["labels"] = [2 for _ in range(len(n_pca_dt))]
        return n_df

    def generate_data(self):
        # If file not exists, create it
        phising_df = self.get_phishing_df()
        non_phishing_df = self.get_non_phishing_df()

        # combine to one df
        df = pd.concat([phising_df, non_phishing_df], ignore_index=True)

        return df

    def get_data(self, path=None):
        # check if file exists
        if os.path.exists(path):
            return True, pd.read_excel(path)
        # if not exist, generate data
        return False, self.generate_data()

    def run(self, save_file_path=None, replace_existing=True):
        path = save_file_path
        if not path:
            path = config.SAVE_PREP_PATH

        is_existing_data, data = self.get_data(path)

        # If data not exist yet or
        # replace_existing option is True,
        # then export data to excel
        if not is_existing_data or replace_existing:
            try:
                data.to_excel(path)
                print("[INFO] Preprocessed data has been saved successfully")
            except IOError:
                print("[ERROR] Cannot save data to excel!")

        if is_existing_data:
            print("[INFO] Using existing data...")

        return data
