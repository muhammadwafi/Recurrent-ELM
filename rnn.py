import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import (
    Dense,
    SimpleRNN,
    Embedding,
    TextVectorization,
    Bidirectional
)
from keras.models import Sequential
from keras.callbacks import EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


data = pd.read_excel("./prep/PCA_data.xlsx")
X = data.loc[:, data.columns.str.startswith("X")]
# X = data["data"]
y = data["labels"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# VOCAB_SIZE = 1000
# BUFFER_SIZE = 10000
# BATCH_SIZE = 64

# encoder = TextVectorization(max_tokens=VOCAB_SIZE)
# encoder.adapt(X_train.tolist())
# vocab = np.array(encoder.get_vocabulary())
# print(vocab[:20])

# model = Sequential([
#     encoder,
#     Embedding(
#         input_dim=len(encoder.get_vocabulary()),
#         output_dim=64,
#         mask_zero=True
#     ),
#     Bidirectional(SimpleRNN(64)),
#     Dense(64, activation='relu'),
#     Dense(1)
# ])

# print([layer.supports_masking for layer in model.layers])

# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     metrics=['accuracy']
# )

# history = model.fit(
#     x=X_train,
#     y=y_train,
#     epochs=10,
#     validation_data=X_test,
#     validation_steps=30
# )

model = Sequential()
model.add(Embedding(1000, 2, input_length=10))
model.add(SimpleRNN(24, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Model is trained and validated for test dataset with 100 epochs.
# Callback is made at an early stage when the validation loss
# has its first minimum value.
early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=10
)

# fit the model
model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    validation_data=(X_test, y_test), verbose=1,
    callbacks=[early_stop]
)

# preds = (model.predict(X_test) > 0.5).astype("int32")
# print(classification_report(y_test, preds, zero_division=0))
preds = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, preds, zero_division=0))
print(preds.shape)
