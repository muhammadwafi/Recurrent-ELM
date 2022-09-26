PHISHING_PATH = "./dataset/phishing"
NON_PHISHING_PATH = "./dataset/non_phishing"

TFIDF_SAVE_PATH = "./prep/TFIDF_data.xlsx"
PCA_SAVE_PATH = "./prep/PCA_data.xlsx"

CLEANED_SAVE_PATH = "./prep/cleaned_data.xlsx"
TOKENIZED_SAVE_PATH = "./prep/tokenizing_data.xlsx"
STOPWORD_SAVE_PATH = "./prep/stopword_data.xlsx"
STEMMED_SAVE_PATH = "./prep/stemming_data.xlsx"

DATA_INFO_PATH = "./results/dataset_info.json"
BEST_PARAMS_PATH = "./results/best_params.json"

TRAIN_SAVE_PATH = "./prep/train_data.xlsx"
TEST_SAVE_PATH = "./prep/test_data.xlsx"

LABEL_ENCODER = {
    "phishing": 1,
    "non_phishing": 2
}

LABEL_DECODER = dict(
    zip(LABEL_ENCODER.values(), LABEL_ENCODER.keys())
)
