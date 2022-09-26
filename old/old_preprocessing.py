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

_tuple_name = [
    'cleaned_html_dt',
    'case_folded_dt',
    'cleaned_dt',
    'stopword_stem_dt',
    'final_dt'
]
_Record = collections.namedtuple('_Record', _tuple_name)


class Preprocessing:
    """Sets of preprocessing functions"""

    def __init__(
        self,
        phishing_path,
        non_phishing_path,
        replace_existing=False,
        use_pca=False,
        limit_data=None
    ):
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
        self.replace_existing = replace_existing
        self.phising_label = 1
        self.non_phishing_label = 2
        self.use_pca = use_pca
        self.limit_data = limit_data
        self.filenames = []

    def case_folding(self, item):
        return item.lower()

    def cleaning(self, item):
        url_rgx = r"((http://[^\s]+)|(https://[^\s]+)|(pic.[^\s]+))"

        # remove new line
        words = re.sub("=", "", item)
        words = re.sub("[\n]", " ", item)
        # remove urls
        words = re.sub(url_rgx, "", words)
        # remove numbers
        words = re.sub(r"(\s|^)?[0-9]+(\s|$)?", " ", words)
        # remove punctutation marks
        words = re.sub(r"[^\w\s]", " ", words)
        # remove spaces
        words = re.sub(r"[\s]+", " ", words)
        # remove double characters if > 2
        words = re.compile(r"(.)\1{2,}").sub(r"\1\1", words)
        return re.sub("[^a-zA-Z]", " ", words)

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
        return BeautifulSoup(item, "html.parser").get_text()

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
        for path in paths:
            print("[INFO] Reading file", path)
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
        cleaned_html_dt = []
        case_folded_dt = []
        cleaned_dt = []
        final_dt = []

        for mail_body in mail_dt:
            cleaned_html = self.remove_html_tag(mail_body)
            cleaned_html_dt.append(cleaned_html)

            case_folded = self.case_folding(cleaned_html)
            case_folded_dt.append(case_folded)

            cleaned = self.cleaning(case_folded)
            cleaned_dt.append(cleaned)

            stopword_removed_and_stemmed = self.stopwords_and_stemming(cleaned)
            final_dt.append(stopword_removed_and_stemmed)

        return {
            "CLEANED_HTML": cleaned_html_dt,
            "CASE_FOLDING": case_folded_dt,
            "CLEANED": cleaned_dt,
            "STOPWORD_STEMMING": final_dt
        }

    def get_tf_idf(self, dataset):
        fitur = self.get_fitur(dataset)
        result = TfIdf(dataset, fitur).getTfIdf()
        # ----- NEW ------
        # word_vectorizer = self.tfidf_vect
        # word_vectorizer.fit(fitur)
        # result = word_vectorizer.transform(fitur)

        return result

    def get_prep_data(self, dataset):
        tfidf = self.get_tf_idf(dataset)
        if self.use_pca:
            pca_obj = PCA(n_components=10, random_state=42)
            # pca_obj = TruncatedSVD(n_components=10, random_state=42)
            pca_obj.fit(tfidf)
            return pca_obj.transform(tfidf)

        return tfidf

    def get_phishing_df(self):
        data = self._get_cleaned_data(self.phishing_path, "phising")

        p_pca_dt = self.get_prep_data(data["STOPWORD_STEMMING"])

        cols = [f"X{i+1}" for i in range(len(p_pca_dt[0]))]
        prep_col_name = ["data"]
        labels = [self.phising_label for _ in range(len(p_pca_dt))]

        # Create phising data dataframe
        p_df = pd.DataFrame(
            data=p_pca_dt,
            columns=cols
        )
        # Cleaned html data frame
        cleaned_html_df = pd.DataFrame(
            data=data["CLEANED_HTML"],
            columns=prep_col_name
        )
        # Case folded data frame
        case_folded_df = pd.DataFrame(
            data=data["CASE_FOLDING"],
            columns=prep_col_name
        )
        # Cleaned data frame
        cleaned_df = pd.DataFrame(
            data=data["CLEANED"],
            columns=prep_col_name
        )
        # Stopword Removal and Stemming data frame
        stopword_stem_df = pd.DataFrame(
            data=data["STOPWORD_STEMMING"],
        )
        # Add labels column
        cleaned_html_df["labels"] = labels
        case_folded_df["labels"] = labels
        cleaned_df["labels"] = labels
        stopword_stem_df["labels"] = labels
        p_df["labels"] = labels

        return _Record(
            cleaned_html_df,
            case_folded_df,
            cleaned_df,
            stopword_stem_df,
            p_df
        )

    def get_non_phishing_df(self):
        data = self._get_cleaned_data(
            self.non_phishing_path, "non_phishing")

        n_pca_dt = self.get_prep_data(data["STOPWORD_STEMMING"])

        cols = [f"X{i+1}" for i in range(len(n_pca_dt[0]))]
        prep_col_name = ["data"]
        labels = [self.non_phishing_label for _ in range(len(n_pca_dt))]

        # Create phising data dataframe
        n_df = pd.DataFrame(
            data=n_pca_dt,
            columns=cols
        )
        # Cleaned html data frame
        cleaned_html_df = pd.DataFrame(
            data=data["CLEANED_HTML"],
            columns=prep_col_name
        )
        # Case folded data frame
        case_folded_df = pd.DataFrame(
            data=data["CASE_FOLDING"],
            columns=prep_col_name
        )
        # Cleaned data frame
        cleaned_df = pd.DataFrame(
            data=data["CLEANED"],
            columns=prep_col_name
        )
        # Stopword Removal and Stemming data frame
        stopword_stem_df = pd.DataFrame(
            data=data["STOPWORD_STEMMING"]
        )
        # Add labels column
        cleaned_html_df["labels"] = labels
        case_folded_df["labels"] = labels
        cleaned_df["labels"] = labels
        stopword_stem_df["labels"] = labels
        n_df["labels"] = labels

        return _Record(
            cleaned_html_df,
            case_folded_df,
            cleaned_df,
            stopword_stem_df,
            n_df
        )

    def generate_data(self):
        # If file not exists, create it
        phising_df = self.get_phishing_df()
        non_phishing_df = self.get_non_phishing_df()

        # combine to one dataframe
        cleaned_html_df = pd.concat(
            [phising_df.cleaned_html_dt, non_phishing_df.cleaned_html_dt],
            ignore_index=True
        )
        cleaned_html_df.insert(0, "filename", self.filenames)

        case_folded_df = pd.concat(
            [phising_df.case_folded_dt, non_phishing_df.case_folded_dt],
            ignore_index=True
        )
        case_folded_df.insert(0, "filename", self.filenames)

        cleaned_df = pd.concat(
            [phising_df.cleaned_dt, non_phishing_df.cleaned_dt],
            ignore_index=True
        )
        cleaned_df.insert(0, "filename", self.filenames)

        stopword_stem_df = pd.concat(
            [phising_df.stopword_stem_dt, non_phishing_df.stopword_stem_dt],
            ignore_index=True
        )
        stopword_stem_df.insert(0, "filename", self.filenames)

        final_df = pd.concat(
            [phising_df.final_dt, non_phishing_df.final_dt],
            ignore_index=True
        )
        final_df.insert(0, "filename", self.filenames)

        save_lists = {
            "CLEANED_HTML": {
                "data": cleaned_html_df,
                "path": config.CLEAN_HTML_SAVE_PATH,
            },
            "CASE_FOLDING": {
                "data": case_folded_df,
                "path": config.CASE_FOLD_SAVE_PATH,
            },
            "CLEANED": {
                "data": cleaned_df,
                "path": config.CLEANED_SAVE_PATH,
            },
            "STOPWORD_STEMMING": {
                "data": stopword_stem_df,
                "path": config.STOPWORD_SAVE_PATH,
            },
        }

        for _, value in save_lists.items():
            if self.replace_existing or not os.path.exists(value['path']):
                self.save(value['path'], value['data'])

        return final_df

    def get_data(self):
        final_df_path = config.TFIDF_SAVE_PATH
        if self.use_pca:
            final_df_path = config.PCA_SAVE_PATH

        if self.replace_existing or not os.path.exists(final_df_path):
            data = self.generate_data()
            self.save(final_df_path, data)

        return pd.read_excel(final_df_path)

    def save(self, path, data):
        print(f"[INFO] Saving data to {path}")
        try:
            data.to_excel(path)
            print(f"[INFO] Data has been saved at {path}")
        except IOError:
            print(f"[ERROR] Cannot save data to {path}!")

        return data, path

    def run(self):
        if self.replace_existing:
            print("[INFO] Preprocessing with replace existing options True...")

        return self.get_data()
