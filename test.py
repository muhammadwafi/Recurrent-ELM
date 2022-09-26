import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from elm import ELMClassifier
import config

data = pd.read_excel("./prep/PCA_data.xlsx")
X = data.loc[:, data.columns.str.startswith("X")]
y = data["labels"]

# X_train, X_test, y_train, y_test = StratifiedKFold(
#     X, y, test_size=0.2, random_state=24, stratify=y
# )

# defining parameter range 
param_grid = {
    "hidden_node": [2, 4, 8],
    "activation": [6, 12, 1],
}

# def split(X, y):
#     kf = StratifiedKFold(n_splits=5, shuffle=True)
#     for train_index, test_index in kf.split(X, y):
#         # set train test
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]


node = []
activation = []
max = 0

for k in tuple(param_grid.keys()):
    if not param_grid[k] > max:
        max = param_grid[k]
    param_grid.pop(k)

print(max)
