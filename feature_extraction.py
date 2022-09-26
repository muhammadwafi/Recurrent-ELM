from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    def __init__(
        self,
        use_pca=True,
        pca_components=10,
        replace_existing=False,
        random_state=42
    ):
        self.use_pca = use_pca
        self.replace_existing = replace_existing
        self.random_state = random_state
        self.tfidf_vect = TfidfVectorizer(
            min_df=1,
            sublinear_tf=False,
            ngram_range=(1, 1),
            smooth_idf=False,
            norm=None
        )
        self.pca_obj = PCA(
            n_components=pca_components,
            svd_solver="arpack",
            random_state=self.random_state,
        )

    def extract_tfidf(self, X_train, X_test):
        word_vectorizer = self.tfidf_vect
        # fit and transform on the train features
        word_vectorizer.fit(X_train)
        X_train = word_vectorizer.transform(X_train)
        # transform test features to sparse matrix
        X_test = word_vectorizer.transform(X_test)

        return X_train, X_test
  
    def extract_pca(self, X_train, X_test):
        tfidf_train, tfidf_test = self.extract_tfidf(X_train, X_test)
        self.pca_obj.fit(tfidf_train)

        X_train = self.pca_obj.transform(tfidf_train)
        X_test = self.pca_obj.transform(tfidf_test)

        return X_train, X_test

    def get_or_save(self):
        pass

    def save(self, path, data):
        print(f"[INFO] Saving data into -> {path}", end=" \t")
        try:
            data.to_excel(path, index_label="No")
            print("[ DONE ]")
        except IOError:
            print(f"[ ERROR ]")

        return data, path
