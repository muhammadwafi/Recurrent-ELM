import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted
)
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import activation_function as af


# define the default activation function list available
AVAILABLE_AF = [
    "sigmoid",
    "hyperbolic_tanh",
    "relu",
    "leaky_relu",
    "linear",
    "cosine",
    "gaussian"
]


class ELMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        hidden_node=2,
        activation="sigmoid",
        context_neurons=2,
        mode="basic",
        rand_weights=None,
        bias=None,
        random_state=None
    ):
        if activation not in AVAILABLE_AF:
            raise ValueError(
                f"Activation method must be one of {', '.join(AVAILABLE_AF)}!"
            )

        # basic elm params
        self.hidden_node = hidden_node
        self.random_state = random_state
        self.activation_method = activation
        self.rand_weights = rand_weights
        self.bias = bias
        # Set mode options to either
        # `standard` or `recurrent`
        self.mode = mode
        # recurrent params
        self.context_neurons = context_neurons

    def calc_h_init(self, values, weights, bias) -> np.ndarray:
        h_init = np.dot(values, weights.T) + bias
        return h_init

    def activation(self, h_init) -> np.ndarray:
        if self.activation_method == "sigmoid":
            result = af.sigmoid(h_init)
        elif self.activation_method == "hyperbolic_tanh":
            result = af.hyperbolic_tanh(h_init)
        elif self.activation_method == "relu":
            result = np.maximum(0, h_init)
        elif self.activation_method == "leaky_relu":
            result = np.maximum(0.01*h_init, h_init)
        elif self.activation_method == "linear":
            result = h_init
        elif self.activation_method == "cosine":
            result = af.cosine(h_init)
        elif self.activation_method == "gaussian":
            result = af.gaussian(h_init, self.rand_weights, self.bias)
        else:
            raise ValueError("Activation function is not supported!")
        return result

    def calc_moore_penrose(self, matrix_h) -> np.ndarray:
        h_inverse = np.linalg.pinv(np.dot(matrix_h.T, matrix_h))
        h_plus = h_inverse.dot(matrix_h.T)
        return h_plus

    def calc_matrix_beta(self, class_data, h_plus):
        class_regroup = pd.get_dummies(class_data)
        matrix_beta = h_plus.dot(class_regroup)
        return matrix_beta

    def calc_output_layer(self, matrix_h, matrix_beta):
        output = matrix_h.dot(matrix_beta)
        return output

    def calc_delay(self, seq: int, n: int, r: int) -> np.ndarray:
        delta = seq - (n + r) + n
        return delta

    def get_matrix_delay(self, X, labels) -> pd.DataFrame:
        matrix_delay = []
        n_features = X.shape[1]
        # Change labels type from df to list
        if not isinstance(labels, list):
            labels = list(labels)

        for r in range(1, self.context_neurons+1):
            seq = 1
            temp_res = []
            for _ in labels:
                # Get result for matrix delay
                res = self.calc_delay(seq, n_features, r)
                temp_res.append(0 if res <= 0 else labels[res])
                seq += 1
            # Append to matrix delay list
            matrix_delay.append(temp_res)

        # Reshape matrix to r x X[rows] and transform
        reshaped = np.reshape(matrix_delay, (r, X.shape[0])).T
        # Convert to DataFrame to join with X
        delay_df = pd.DataFrame(
            reshaped,
            columns=[f"Xd{i}" for i in range(1, self.context_neurons+1)]
        )

        return delay_df

    def score(self, y_pred, y_test, with_percentage=False) -> float:
        result = accuracy_score(y_test, y_pred)
        if with_percentage:
            result = result * 100
        return result

    def get_fscore(self, y_pred, y_test,
                   score_type="f1_score",
                   average="binary") -> float:
        if score_type == "f1_score":
            check = f1_score(y_test, y_pred, average=average)
        elif score_type == "recall_score":
            check = recall_score(y_test, y_pred, average=average)
        elif score_type == "precision_score":
            check = precision_score(y_test, y_pred, average=average)
        else:
            check = accuracy_score(y_test, y_pred)
        return check

    def get_join_matrix_delay(self, X, labels):
        matrix_delay = self.get_matrix_delay(X, labels)
        # Rest index and join data (X) with matrix delay
        X = X.reset_index(drop=True)
        X = X.join(matrix_delay)
        return X

    def fit(self, X, labels):
        # Check that X and y have correct shape
        base_X, base_y = check_X_y(X, labels)
        # Store the classes seen during fit
        self.classes_ = unique_labels(base_y)

        if self.mode == "recurrent":
            X = self.get_join_matrix_delay(X, labels)

        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)

        if self.bias is None:
            self.bias = np.random.uniform(
                0.0, 1.0, (1, self.hidden_node))

        if self.rand_weights is None:
            self.rand_weights = np.random.uniform(
                -1.0, 1.0, (self.hidden_node, X.shape[1]))

        bias = self.bias
        if self.mode == "recurrent":
            bias = np.repeat(self.bias, len(X), axis=0)

        # Calc h_init
        h_init = self.calc_h_init(X, self.rand_weights, bias)
        # Activation function
        matrix_h = self.activation(h_init)
        # set matrix_h var as global _matrix_h_train
        # to get the train result
        self._matrix_h_train = matrix_h
        self._train_labels = labels
        # Moore penrose
        h_plus = self.calc_moore_penrose(matrix_h)
        # Matrix Beta
        self.matrix_beta = self.calc_matrix_beta(labels, h_plus)

        self.X_ = base_X
        self.y_ = base_y

        return self

    def get_train_result(self, get_score=False):
        # Output layer
        output_layer = self.calc_output_layer(
            self._matrix_h_train, self.matrix_beta)
        # Predicting
        predict_train = np.argmax(output_layer, axis=1) + 1
        # get train accuracy
        train_accuracy = self.score(self._train_labels, predict_train)
        # if get score is true, return train score
        # else return predict labels
        result = train_accuracy if get_score else predict_train
        return result

    def predict(self, X, labels=[]):
        # Check if fit has been called
        check_is_fitted(self)
        check_array(X)

        if self.mode == "recurrent":
            X = self.get_join_matrix_delay(X, labels)

        bias = self.bias
        if self.mode == "recurrent":
            bias = np.repeat(self.bias, len(X), axis=0)

        # Calculate h_init
        h_init = self.calc_h_init(X, self.rand_weights, bias)
        # Activation function
        matrix_h = self.activation(h_init)
        # Output layer
        output = self.calc_output_layer(matrix_h, self.matrix_beta)
        y_pred = np.argmax(output, axis=1) + 1

        return y_pred

    def get_rand_weights(self) -> np.ndarray:
        # return random weights
        return self.rand_weights

    def get_bias(self) -> np.ndarray:
        # return bias
        return self.bias
