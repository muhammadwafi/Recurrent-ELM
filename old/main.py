import preprocessing
from tfidf import TfIdf
from sklearn.decomposition import PCA

if __name__ == "__main__":
    phishing_path = "./dataset/phishing"
    non_phishing_path = "./dataset/non_phishing"
    file_phising_list = preprocessing.get_file_names(phishing_path, "eml", 5)
    file_non_phising_list = preprocessing.get_file_names(
        non_phishing_path, None, 5)
    print(file_phising_list)
    # print(file_non_phising_list)

    phishing_body_emails = preprocessing.open_email(file_phising_list)
    non_phishing_body_emails = preprocessing.open_email(file_non_phising_list)
    clean_data_phishing = list()
    clean_data_non_phishing = list()
    # preprocessing
    for phishing_body_email in phishing_body_emails:
        cleaned_html = preprocessing.remove_html_tag(phishing_body_email)
        case_folded = preprocessing.case_folding(cleaned_html)
        cleaned = preprocessing.cleaning(case_folded)
        stopword_removed_and_stemmed = preprocessing.stopwords_removal_and_stemming(
            cleaned)
        clean_data_phishing.append(stopword_removed_and_stemmed)
    # print(clean_data_phishing)

    for non_phishing_body_email in non_phishing_body_emails:
        cleaned_html = preprocessing.remove_html_tag(non_phishing_body_email)
        case_folded = preprocessing.case_folding(cleaned_html)
        cleaned = preprocessing.cleaning(case_folded)
        stopword_removed_and_stemmed = preprocessing.stopwords_removal_and_stemming(
            cleaned)
        clean_data_non_phishing.append(stopword_removed_and_stemmed)
    # print(clean_data_non_phishing)

    dataset = clean_data_phishing + clean_data_non_phishing
    fitur = preprocessing.get_fitur(dataset)
    # print("==> fitur ==>", fitur)
    tfidf_obj = TfIdf(dataset, fitur)
    tfidf = tfidf_obj.getTfIdf()
    # print(tfidf)

    # pca
    pca_obj = PCA(n_components=10)
    pca_obj.fit(tfidf)
    pca_result = pca_obj.transform(tfidf)
    print(pca_result)
