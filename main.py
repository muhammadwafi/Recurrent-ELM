from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing
from recurrent_elm import RELMClassifier
import config


def measure_time(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


if __name__ == "__main__":
    phishing_path = config.PHISHING_PATH
    non_phishing_path = config.NON_PHISHING_PATH

    prep = Preprocessing(
        phishing_path=phishing_path,
        non_phishing_path=non_phishing_path
    )

    df = prep.run(config.SAVE_PREP_PATH, replace_existing=False)
    X, y = df.loc[:, df.columns.str.startswith('X')], df["labels"]

    # Split to train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=24)

    relm_options = {
        "hidden_node": 2,
        "activation_method": "sigmoid",
        "context_neurons": 2,
    }

    # Start measure time
    start = timer()

    # Define RELM classifier
    elm = RELMClassifier(**relm_options)
    # fit/train elm
    elm.fit(X_train, y_train)
    train_accuracy = elm.get_train_result(get_score=True)
    # predict data
    y_pred = elm.predict(X_test, y_test)
    # get accuracy
    test_accuracy = elm.score(y_pred, y_test)

    end = timer()
    elapsed_time = measure_time(start, end)

    print(40*"-")
    print("Train accuracy \t:", train_accuracy)
    print("Test accuracy \t:", test_accuracy)
    print("Time elapsed \t:", elapsed_time)
    print(40*"-")
