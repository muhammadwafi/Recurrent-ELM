import os
import json
import itertools
from typing import Callable
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from preprocessing import Preprocessing
from elm import ELMClassifier
from export_helper import export_json, export_data, export_multiframes
import config


def measure_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


class Thesis:
    def __init__(
        self,
        data,
        classifier={},
        test_size=0.2,
        random_state=24,
        replace_existing=False,
        save=False
    ):
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.replace_existing = replace_existing
        self.classifier = classifier
        self.save = save

        self.X = self.data.loc[:, self.data.columns.str.startswith("X")]
        self.y = self.data["labels"]

    # ----- Initial data split
    # @return: tuple()
    # @desc  : return a tuple of dataframe
    #        : for data_train and data_test
    def split_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        if not os.path.exists(config.TRAIN_SAVE_PATH):
            train_df = self.data.iloc[X_train.index]
            print(export_data(train_df, config.TRAIN_SAVE_PATH))

        if not os.path.exists(config.TEST_SAVE_PATH):
            test_df = self.data.iloc[X_test.index]
            print(export_data(test_df, config.TEST_SAVE_PATH))

        return X_train, X_test, y_train, y_test

    def get_count_labels(self, data):
        return data.replace(config.LABEL_DECODER).groupby("labels").size()

    def describe_dataset(self, train_index, test_index):
        if self.replace_existing or not os.path.exists(config.DATA_INFO_PATH):
            all_labels = self.get_count_labels(
                data=self.data
            )
            train_labels = self.get_count_labels(
                data=self.data.iloc[train_index]
            )
            test_labels = self.get_count_labels(
                data=self.data.iloc[test_index]
            )
            desc = {
                "all_data": {
                    "phishing": int(all_labels["phishing"]),
                    "non_phishing": int(all_labels["non_phishing"]),
                    "total_data": int(self.data.shape[0]),
                },
                "train_data": {
                    "phishing": int(train_labels["phishing"]),
                    "non_phishing": int(train_labels["non_phishing"]),
                    "total": int(self.data.iloc[train_index].shape[0]),
                },
                "test_data": {
                    "phishing": int(test_labels["phishing"]),
                    "non_phishing": int(test_labels["non_phishing"]),
                    "total": int(self.data.iloc[test_index].shape[0]),
                },
            }
            export_json(config.DATA_INFO_PATH, desc)

        print(50 * "=")
        print("DATA INFO")
        print(50 * "=")
        with open(config.DATA_INFO_PATH, "r") as data_file:
            load_file = json.load(data_file)
            print(json.dumps(load_file, indent=4))
        print(50 * "=", end="\n\n")

        return True

    # ----- ELM
    # @return: prediction labels and true labels
    # @desc  : return a tuple of y_prediction and y_labels
    def _elm(self, X_train, X_test, y_train, y_test, **params):
        try:
            # unpack params
            hidden_node, activation = (
                params["hidden_node"],
                params["activation"]
            )
        except BaseException:
            raise KeyError("Missing hidden_node and or activation type!")
        elm = ELMClassifier(
            mode="basic", hidden_node=hidden_node, activation=activation
        )
        elm.fit(X_train, y_train)
        y_pred = elm.predict(X_test, y_test)
        return y_pred, y_test

    # ----- Recurrent ELM
    # @return: prediction labels and true labels
    # @desc  : return a tuple of y_prediction and y_labels
    def _recurrent_elm(self, X_train, X_test, y_train, y_test, **params):
        try:
            # unpack params
            hidden_node, activation, context_neurons = (
                params["hidden_node"],
                params["activation"],
                params["context_neurons"],
            )
        except BaseException:
            raise KeyError(
                "Missing hidden_node,activation and context_neurons!")

        relm = ELMClassifier(
            mode="recurrent",
            hidden_node=hidden_node,
            activation=activation,
            context_neurons=context_neurons,
        )
        relm.fit(X_train, y_train)
        y_pred = relm.predict(X_test, y_test)
        return y_pred, y_test

    # ----- Naive Bayes
    # @return: prediction labels and true labels
    # @desc  : return a tuple of y_prediction and y_labels
    def _naive_bayes(self, X_train, X_test, y_train, y_test, **params):
        try:
            # unpack params
            var_smoothing = params["var_smoothing"]
        except BaseException:
            raise KeyError("Missing var_smoothing params!")

        naive_bayes = GaussianNB(var_smoothing=var_smoothing)
        naive_bayes.fit(X_train, y_train)
        y_pred = naive_bayes.predict(X_test)
        return y_pred, y_test

    # ----- Get Classifier
    # @return: Tuple of y_pred and y_true
    # @desc  : return a tuple of y_pred and y_true
    #          result of classifier
    def _get_classifier(
        self,
        cf_name,
        X_train,
        y_train,
        X_test,
        y_test,
        **params,
    ):
        data = (X_train, X_test, y_train, y_test)
        y_pred, y_true = [], []

        cf_list = ["ELM", "RecurrentELM", "NaiveBayes", "RNN"]
        if cf_name not in cf_list:
            raise KeyError(f"{cf_name} must be one of {', '.join(cf_list)}")

        # Get classifier results
        if cf_name == "ELM":
            y_pred, y_true = self._elm(*data, **params)

        if cf_name == "RecurrentELM":
            y_pred, y_true = self._recurrent_elm(*data, **params)

        if cf_name == "NaiveBayes":
            y_pred, y_true = self._naive_bayes(*data, **params)

        return y_pred, y_true

    # ----- Get acc and other metrics
    # @return: dict()
    # @desc  : return pandas dict of
    #          metrics for classifier
    def get_metrics(self, y_pred, y_true):
        # get unique labels
        labels = list(set(y_true))
        metrics_options = {
            "y_true": y_true,
            "y_pred": y_pred,
            "zero_division": 0,
            "labels": labels,
        }

        accuracy = accuracy_score(y_true, y_pred)
        # f1 score
        f1_macro = f1_score(**metrics_options, average="macro")
        f1_micro = f1_score(**metrics_options, average="micro")
        # precision
        prec_macro = precision_score(**metrics_options, average="macro")
        prec_micro = precision_score(**metrics_options, average="micro")
        # recall
        recall_macro = recall_score(**metrics_options, average="macro")
        recall_micro = recall_score(**metrics_options, average="micro")
        # Get confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        # Append to res dict
        result = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision_macro": prec_macro,
            "precision_micro": prec_micro,
            "recall_macro": recall_macro,
            "recall_micro": recall_micro,
        }
        return result, cf_matrix

    # ----- Get best parameters through KFold
    # @return: tuple()
    # @desc  : return raw_result (dict) per fold
    #          and mean_result (float)
    def _kfold_split(self, X, y, method: Callable, fold=5, **params):
        # Stratified kfold
        kf = StratifiedKFold(
            n_splits=fold, random_state=self.random_state, shuffle=True
        )
        elm_params = {"hidden_node": [], "activation": []}
        nb_params = {"var_smoothing": []}

        relm_params = {
            "hidden_node": [],
            "activation": [],
            "context_neurons": []
        }

        res = {
            "accuracy": [],
            "f1_macro": [],
            "f1_micro": [],
            "precision_macro": [],
            "precision_micro": [],
            "recall_macro": [],
            "recall_micro": [],
            "time_elapsed": []
        }

        temp_accuracy = []
        fold_counter = 1

        for train_index, test_index in kf.split(X, y):
            # set train test
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # start record time
            start = timer()
            # get label prediction and actual label from classifier
            y_pred, y_true = method(X_train, X_test, y_train, y_test, **params)
            result_metrics, _ = self.get_metrics(y_pred, y_true)
            # end record time
            end = timer()
            elapsed_time = timedelta(seconds=end - start).total_seconds()
            # add time elapsed
            result_metrics.update({"time_elapsed": float(elapsed_time)})

            # Append result
            for metric in res.keys():
                res[metric].append(result_metrics[metric])

            # add params to res dict
            if method.__name__ == "_elm":
                elm_params["hidden_node"].append(params["hidden_node"])
                elm_params["activation"].append(params["activation"])
            elif method.__name__ == "_recurrent_elm":
                relm_params["hidden_node"].append(params["hidden_node"])
                relm_params["activation"].append(params["activation"])
                relm_params["context_neurons"].append(
                    params["context_neurons"]
                )
            elif method.__name__ == "_naive_bayes":
                nb_params["var_smoothing"].append(params["var_smoothing"])
            else:
                raise ValueError("Function does not exists!")

            temp_accuracy.append(result_metrics["accuracy"])

            print(
                " F{} └── Accuracy: {:.2f}% -> Time: {}".format(
                    fold_counter, result_metrics["accuracy"], elapsed_time
                )
            )

            fold_counter += 1

        mean_accuracy = np.array(temp_accuracy).mean()
        print(f"├── Kfold mean accuracy: {mean_accuracy:.2f}%")
        print("│")

        if method.__name__ == "_elm":
            res.update(elm_params)
        elif method.__name__ == "_recurrent_elm":
            res.update(relm_params)
        elif method.__name__ == "_naive_bayes":
            res.update(nb_params)
        else:
            raise ValueError("Function not exists!")

        # return raw_result and mean_acc
        return res, mean_accuracy

    # ----- Predict using cross validation
    # @return: dict()
    # @desc  : return best_params dict() for
    #          each classifier
    def _get_best_params(self, X, y, fold=5, category=None):
        best_params = {cf: [] for cf, _ in self.classifier.items()}
        metrics = [
            "accuracy",
            "f1_macro",
            "f1_micro",
            "precision_macro",
            "precision_micro",
            "recall_macro",
            "recall_micro",
            "time_elapsed",
        ]
        # Iterate over classifier and its params
        for cf_name, params in self.classifier.items():
            kf_results, df_title = [], []
            print(f"│\n├── Predicting {cf_name} method:")
            # Iterate over fold to get best params
            if cf_name == "ELM":
                method_params = list(itertools.product(
                    params["hidden_node"],
                    params["activation"]
                ))
                for p in method_params:
                    elm_param = {"hidden_node": p[0], "activation": p[1]}
                    print(
                        f"├── KFold for Node: {p[0]} & Activation: {p[1]}:"
                    )
                    elm_result, elm_mean = self._kfold_split(
                        X, y, self._elm, fold=fold, **elm_param
                    )
                    elm_df = pd.DataFrame.from_dict(elm_result)
                    # calculate mean
                    elm_df.loc["mean"] = elm_df[metrics].mean()
                    # append results
                    kf_results.append(elm_df)
                    df_title.append(f"Node: {p[0]} | Activation: {p[1]}")
                    # append best params
                    best_params[cf_name].append({
                        "mean_acc": elm_mean,
                        **elm_param
                    })

            elif cf_name == "RecurrentELM":
                method_params = list(itertools.product(
                    params["hidden_node"],
                    params["activation"],
                    params["context_neurons"]
                ))
                for p in method_params:
                    relm_param = {
                        "hidden_node": p[0],
                        "activation": p[1],
                        "context_neurons": p[2]
                    }
                    print(
                        f"├── KFold for Node: {p[0]}, Activation: {p[1]} & ContextNeurons: {p[2]}"
                    )
                    relm_result, relm_mean = self._kfold_split(
                        X, y, self._recurrent_elm, fold=fold, **relm_param
                    )
                    relm_df = pd.DataFrame.from_dict(relm_result)
                    # calculate mean
                    relm_df.loc["mean"] = relm_df[metrics].mean()
                    # append results
                    kf_results.append(relm_df)
                    df_title.append(f"Node: {p[0]} | Activation: {p[1]} | ContextNeurons: {p[2]}")
                    # append best params
                    best_params[cf_name].append({
                        **relm_param,
                        "mean_acc": relm_mean
                    })

            elif cf_name == "NaiveBayes":
                for p in params["var_smoothing"]:
                    nb_param = {"var_smoothing": p}
                    print(f"├── KFold for var_smoothing: {p}")
                    nb_result, nb_mean = self._kfold_split(
                        X, y, self._naive_bayes, fold=fold, **nb_param
                    )
                    nb_df = pd.DataFrame.from_dict(nb_result)
                    # calculate mean
                    nb_df.loc["mean"] = nb_df[metrics].mean()
                    # append results
                    kf_results.append(nb_df)
                    df_title.append(f"VarSmoothing: {p}")
                    # append best params
                    best_params[cf_name].append({
                        **nb_param,
                        "mean_acc": nb_mean
                    })

            else:
                raise KeyError(
                    f"{cf_name} must be one of [ELM, RecurrentELM, NaiveBayes]"
                )

            if self.save:
                export_multiframes(
                    f"./results/raw/{cf_name}_BestParams.xlsx",
                    datas=kf_results,
                    df_title=df_title,
                )

        return best_params


    def _predict(self, X_train, X_test, y_train, y_test, **best_params):
        res = {
            "classifier"     : [],
            "accuracy"       : [],
            "f1_macro"       : [],
            "f1_micro"       : [],
            "precision_macro": [],
            "precision_micro": [],
            "recall_macro"   : [],
            "recall_micro"   : [],
            "time_elapsed"   : [],
            "y_pred"         : [],
            "y_true"         : [],
            "labels"         : [],
        }
        cf_matrix_result = {
            "classifier": [],
            "confusion_matrix": [],
            "y_labels": []
        }
        print(45*"#")
        # Iterate over classifier and its params
        for cf_name, params in best_params.items():
            start = timer()
            y_pred, y_true = self._get_classifier(
                cf_name,
                X_train,
                y_train,
                X_test,
                y_test,
                **params
            )
            result, cf_matrix = self.get_metrics(y_pred, y_true)
            end = timer()
            elapsed_time = timedelta(seconds=end - start).total_seconds()\
    
            # Append result
            for metric in result.keys():
                res[metric].append(result[metric])

            labels = list(set(y_test))
            res["classifier"].append(cf_name)
            res["labels"].append(labels)
            res["y_pred"].append(y_pred)
            res["y_true"].append(y_true)
            res["time_elapsed"].append(float(elapsed_time))

            # append confusion matrix result
            cf_matrix_result["classifier"].append(cf_name)
            cf_matrix_result["confusion_matrix"].append(cf_matrix)
            cf_matrix_result["y_labels"].append([
                config.LABEL_DECODER.get(s) for s in labels
            ])

            print("├── {}: {:.2f}% -> Time: {}"
                  .format(cf_name, result["accuracy"]*100, elapsed_time))

        if self.save:
            try:
                res_df = pd.DataFrame.from_dict(res)
                res_df = res_df.drop(['y_pred', 'y_true', 'labels'], axis=1)
                export_data(res_df, f"./results/final_results.xlsx")

                # Save Confusion Matrix
                for n_cf in range(len(cf_matrix_result["classifier"])):
                    cf_index = [
                        '(true):{:}'.format(x) for x in cf_matrix_result["y_labels"][n_cf]
                    ]
                    cf_columns = [
                        '(pred):{:}'.format(x) for x in cf_matrix_result["y_labels"][n_cf]
                    ]
                    cf_filename = f"{cf_matrix_result['classifier'][n_cf]}_ConfusionMatrix"

                    cf_matrix_df = pd.DataFrame(
                        cf_matrix_result["confusion_matrix"][n_cf],
                        index=cf_index,
                        columns=cf_columns
                    )
                    export_data(cf_matrix_df, f"./results/{cf_filename}.xlsx")
            except Exception as e:
                print(e)
                print("Cannot save final results to excel!")
            
        return res

    def run(self, train_folds=5, multi_times=False, n_times=0):
        # Splitting dataset
        X_train, X_test, y_train, y_test = self.split_dataset()

        # Describe all data, train data and test data
        self.describe_dataset(X_train.index, X_test.index)

        # Get best params through cross validation
        cv_result = self._get_best_params(
            X_train, y_train, fold=train_folds
        )
        best_params = {}
        for x, param in cv_result.items():
            # Find max score in mean accuracy
            max_mean_acc = max(param, key=lambda x: x['mean_acc'])
            best_params.update({x: max_mean_acc})

        # Get metric using best params on each classifier
        self._predict(X_train, X_test, y_train, y_test, **best_params)

        if self.save:
            try:
                with open(config.BEST_PARAMS_PATH, "w") as file:
                    json.dump(best_params, file, indent=4, sort_keys=True)
            except IOError:
                return ("[ERROR] Cannot export best parameters result!")

        return "\n\n[ DONE ]"


if __name__ == "__main__":
    # Run preprocessing data
    # prep_data = Preprocessing(
    #     phishing_path=config.PHISHING_PATH,
    #     non_phishing_path=config.NON_PHISHING_PATH,
    #     use_pca=False,
    #     n_components=10,
    #     random_state=42,
    #     ngram_range=(1, 1),
    #     replace_existing=False
    # ).run()

    prep_data = pd.read_excel(config.PCA_SAVE_PATH)

    classifier = {
        "ELM": {
            "hidden_node": [2, 4, 8],
            "activation": ["sigmoid", "cosine", "relu"],
        },
        "RecurrentELM": {
            "hidden_node": [2, 4, 8],
            "activation": ["sigmoid", "cosine", "relu"],
            "context_neurons": [2, 4, 8]
        },
        "NaiveBayes": {
            "var_smoothing": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        },
    }

    thesis = Thesis(data=prep_data, save=True, classifier=classifier)
    thesis.run(train_folds=5, multi_times=True, n_times=10)
