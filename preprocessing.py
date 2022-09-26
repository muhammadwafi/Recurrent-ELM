import os
import re
import email
import glob
import collections
from timeit import default_timer as timer
from datetime import timedelta
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import pandas as pd
import config

_tuple_name = [
    "cleaned",
    "tokenized",
    "stopwords",
    "stemmed"
]
_Record = collections.namedtuple("_Record", _tuple_name)


class Preprocessing:
    """Sets of preprocessing functions"""

    def __init__(
        self,
        phishing_path,
        non_phishing_path,
        replace_existing=False,
        limit_data=None,
        use_pca=True,
        n_components=10,
        random_state=42,
        ngram_range=(1, 1),
    ):
        self.phishing_path = phishing_path
        self.non_phishing_path = non_phishing_path
        self.porter_stemmer = PorterStemmer()
        self.clean_regex = re.compile("[^a-zA-Z]")
        self.replace_existing = replace_existing
        self.limit_data = limit_data
        self.filenames = []
        self.stopwords = stopwords.words("english")
        self.n_components = n_components
        self.use_pca = use_pca
        self.tfidf_vect = TfidfVectorizer(
            min_df=1,
            ngram_range=ngram_range,
        )
        self.pca_obj = PCA(
            n_components=n_components,
            random_state=random_state,
            # svd_solver="arpack",
        )

    def cleaning(self, words):
        url_rgx = r"((http://[^\s]+)|(https://[^\s]+)|(pic.[^\s]+))"

        # remove html tags
        words = BeautifulSoup(words, "html.parser").get_text()
        # Case folding
        words = words.lower()
        # remove new line
        words = re.sub("=\n", "", words)
        words = re.sub("[\n]", " ", words)
        # remove urls
        words = re.sub(url_rgx, "", words)
        # remove numbers
        words = re.sub(r"(\s|^)?[0-9]+(\s|$)?", " ", words)
        # remove punctutation marks
        words = re.sub(r"e-mail", "email", words)
        words = re.sub(r"[IMAGE]", "", words)
        words = re.sub(r"[^\w\s]", " ", words)
        # remove spaces
        words = re.sub(r"[\s]+", " ", words)
        # remove double characters if > 2
        words = re.compile(r"(.)\1{2,}").sub(r"\1\1", words)
        words = re.sub("[^a-zA-Z]", " ", words)
        return words.strip()

    def tokenizer(self, cleaned_dt):
        # cleaned = self.cleaning(words)
        return cleaned_dt.split()

    def stopwords_removal(self, token_dt):
        return [
            word for word in token_dt if word not in self.stopwords
        ]

    def stemming(self, stopwords_dt):
        return [
            self.porter_stemmer.stem(word) for word in stopwords_dt
        ]

    def get_file_names(self, path, filetype=None):
        if filetype:
            path += f"/*.{filetype}"
        else:
            path += "/*"

        file_list = glob.glob(path)
        return file_list[:self.limit_data]

    def get_body_email(self, data):
        if type(data) == str:
            return data
        return self.get_body_email(data[0].get_payload())

    def open_email(self, paths):
        emails = []
        self.filenames = []
        for path in paths:
            self.filenames.append(os.path.basename(path))
            try:
                message = email.message_from_file(
                    open(path, encoding="cp1252", errors="ignore"))
                payload = message.get_payload()
                body = self.get_body_email(payload)
                emails.append(body)
            except IOError:
                emails.append("")
        return emails

    def extract_tfidf(self, dataset) -> tuple:
        # transform features to sparse matrix
        tfidf = self.tfidf_vect.fit_transform(dataset)
        # get all feature names
        feature_names = self.tfidf_vect.get_feature_names_out()
        return tfidf, feature_names

    def extract_pca(self, dataset):
        tfidf = self.tfidf_vect.fit_transform(dataset)
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        result = svd.fit_transform(tfidf)
        return result

    def convert_to_dataframe(self, data, labels):
        cleaned_df = pd.DataFrame({
            "filenames": self.filenames,
            "data": data.cleaned,
            "labels": labels,
        })

        tokenized_df = pd.DataFrame({
            "filenames": self.filenames,
            "data": data.tokenized,
            "labels": labels,
        })

        stopword_df = pd.DataFrame({
            "filenames": self.filenames,
            "data": data.stopwords,
            "labels": labels,
        })

        stemmed_df = pd.DataFrame({
            "filenames": self.filenames,
            "data": data.stemmed,
            "labels": labels,
        })

        return _Record(
            cleaned_df, tokenized_df, stopword_df, stemmed_df
        )

    def get_data(self, data_path, data_type):
        file_list = self.get_file_names(data_path, None)
        labels_code = config.LABEL_ENCODER["non_phishing"]
        if data_type == "phishing":
            labels_code = config.LABEL_ENCODER["phishing"]
            file_list = self.get_file_names(data_path, "eml")

        mail_dt = self.open_email(file_list)
        total_file = len(file_list)
        counter = 0

        cleaned_dt = []
        tokenized_dt = []
        stopword_dt = []
        stemmed_dt = []

        for mail_body, path in zip(mail_dt, file_list):
            fname = os.path.basename(path)
            counter += 1
            info = f"{counter}/{total_file}"
            # progress = (complete / total_file) * 100

            print(f"Processing file {fname}", end="\r", flush=True)
            print(f"   [+] Cleaning file {fname}", end="\r", flush=True)
            cleaned_words = self.cleaning(mail_body)
            cleaned_dt.append(cleaned_words)

            print(f"   [+] Tokenizing file {fname}", end="\r", flush=True)
            tokenized_words = self.tokenizer(cleaned_words)
            tokenized_dt.append(tokenized_words)

            print(f"   [+] Stopwords removal {fname}", end="\r", flush=True)
            stopwords_removed = self.stopwords_removal(tokenized_words)
            stopword_dt.append(stopwords_removed)

            print(f"   [+] Stemming file {fname}", end="\r", flush=True)
            stemmed_words = self.stemming(stopwords_removed)
            stemmed_dt.append(stemmed_words)

            print(f"[ OK ] Process complete for -> {path} \t [{info}]")

        print(64*"-")

        total = range(len(mail_dt))
        labels = [labels_code for _ in total]

        return self.convert_to_dataframe(
            data=_Record(
                cleaned_dt, tokenized_dt, stopword_dt, stemmed_dt
            ),
            labels=labels
        )

    def get_joined_dataframe(self):
        phishing = self.get_data(self.phishing_path, "phishing")
        non_phishing = self.get_data(self.non_phishing_path, "non_phishing")

        # Remove index data of pandas
        # default indexes and join them

        # remove `cleaned` index
        phishing.cleaned.reset_index(drop=True, inplace=True)
        non_phishing.cleaned.reset_index(drop=True, inplace=True)
        cleaned_df = pd.concat(
            [phishing.cleaned, non_phishing.cleaned],
            ignore_index=True
        )
        # remove `tokenized` index
        phishing.tokenized.reset_index(drop=True, inplace=True)
        non_phishing.tokenized.reset_index(drop=True, inplace=True)
        tokenized_df = pd.concat(
            [phishing.tokenized, non_phishing.tokenized],
            ignore_index=True
        )
        # remove `stopword` index
        phishing.stopwords.reset_index(drop=True, inplace=True)
        non_phishing.stopwords.reset_index(drop=True, inplace=True)
        stopword_df = pd.concat(
            [phishing.stopwords, non_phishing.stopwords],
            ignore_index=True
        )
        # remove `stemmed` index
        phishing.stemmed.reset_index(drop=True, inplace=True)
        non_phishing.stemmed.reset_index(drop=True, inplace=True)
        stemmed_df = pd.concat(
            [phishing.stemmed, non_phishing.stemmed],
            ignore_index=True
        )

        return {
            "cleaned": cleaned_df,
            "tokenized": tokenized_df,
            "stopword": stopword_df,
            "stemmed": stemmed_df,
        }

    def get_or_save(self):
        path_list = {
            "cleaned": config.CLEANED_SAVE_PATH,
            "tokenized": config.TOKENIZED_SAVE_PATH,
            "stopword": config.STOPWORD_SAVE_PATH,
            "stemmed": config.STEMMED_SAVE_PATH
        }

        # start = timer()

        if self.replace_existing:
            data = self.get_joined_dataframe()

            for name, path in path_list.items():
                self.save(path, data.get(name))

        for name, path in path_list.items():
            if not os.path.exists(path):
                self.save(path, data.get(name))

        # end = timer()
        # elapsed_time = timedelta(seconds=end-start)

        # if self.replace_existing:
        #     print(64*"-")
        #     print(f"[DONE] Preprocessing data completed in {elapsed_time}s")
        #     print(64*"-")
        #     print()

        return pd.read_excel(path_list["stemmed"])

    def get_or_save_tfidf(self, stemmed_df, skip_cols=None):
        if self.replace_existing or not os.path.exists(config.TFIDF_SAVE_PATH):
            print("[INFO] Getting TF-IDF features...", end="\r", flush=True)

            tfidf, feature_names = self.extract_tfidf(stemmed_df["data"])
            # convert to dataframe
            df_tfidf = pd.DataFrame(
                tfidf.todense(),
                columns=[
                    f"X{i+1} ({feature_names[i]})"
                    for i in range(len(feature_names))
                ]
            )
            # append labels
            df_tfidf.insert(0, "filenames", stemmed_df["filenames"])
            df_tfidf["labels"] = stemmed_df["labels"]
            # export data to excel
            self.save(config.TFIDF_SAVE_PATH, df_tfidf)

        if skip_cols:
            return pd.read_excel(
                config.TFIDF_SAVE_PATH,
                usecols=lambda x: x not in skip_cols
            )

        return pd.read_excel(config.TFIDF_SAVE_PATH)

    def get_or_save_pca(self, stemmed_df):
        if self.replace_existing or not os.path.exists(config.PCA_SAVE_PATH):
            print("[INFO] Getting PCA features...", end="\r", flush=True)
            pca = self.extract_pca(stemmed_df["data"])
            df_pca = pd.DataFrame(
                pca,
                columns=[f"X{i+1}" for i in range(len(pca[0]))]
            )
            df_pca.insert(0, "filenames", stemmed_df["filenames"])
            df_pca["labels"] = stemmed_df["labels"]
            print("[ DONE ]")
            # export data to excel
            self.save(config.PCA_SAVE_PATH, df_pca)

        return pd.read_excel(config.PCA_SAVE_PATH)

    def save(self, path, data):
        print(f"[INFO] Saving data into -> {path}", end=" \t")
        try:
            data.to_excel(path, index_label="No")
            print("[ DONE ]")
        except IOError:
            print("[ ERROR ]")

        return data, path

    def run(self):
        path = config.TFIDF_SAVE_PATH
        if self.use_pca:
            path = config.PCA_SAVE_PATH

        if not self.replace_existing:
            print(f"[INFO] Load existing data from -> {path}\n")

        stemmed_df = self.get_or_save()

        if self.use_pca:
            return self.get_or_save_pca(stemmed_df)

        return self.get_or_save_tfidf(stemmed_df)
