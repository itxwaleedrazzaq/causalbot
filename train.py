import os
import pickle

import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                             ReduceLROnPlateau)
from keras._tf_keras.keras.layers import (LSTM, Activation, Conv2D, Dense,
                                          Flatten, GlobalAveragePooling2D,
                                          Input, Lambda, MultiHeadAttention,
                                          TimeDistributed)
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import l2
from keras_nac.layers import LAN, NAC
from ncps.keras import CfC, LTC
from ncps.wirings import AutoNCP
from sklearn.preprocessing import MinMaxScaler

from utils import (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, create_sequences,
                   load_pickle, prepare_data)

DATA = os.path.join("data", "training_data.pkl")
MODEL_NAME = "deepbot"
WEIGHTS_DIR = "model_weights"

TRAIN_SIZE = 0.8

SEQUENCE_LENGTH = 8
BATCH_SIZE = 64

EPOCHS = 200
LEARNING_RATE = 8.5e-05
SEED = 100

# Which recurrent cell to train. One of: NAC, LSTM, LAN, SDPA, LTC, CfC.
TRAIN_POLICY = "LAN"

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
images, labels = load_pickle(DATA)
labels = np.asarray(labels, dtype=np.float32)

# Labels are scaled to [-1, 1] to match the tanh output of most cells; the same
# scaler is pickled next to the weights and inverted by the test node.
scaler = MinMaxScaler(feature_range=(-1, 1))
labels = scaler.fit_transform(labels)
with open(os.path.join(WEIGHTS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("\nCombined dataset")
print("Images:", len(images))
print("Labels:", len(labels))

# Chronological split: the tail of each drive becomes the validation set.
train_end = int(len(images) * TRAIN_SIZE)

print("\nPreprocessing training images...")
train_images, train_labels = prepare_data(images[:train_end], labels[:train_end],
                                          is_training=True)

print("Preprocessing validation images...")
valid_images, valid_labels = prepare_data(images[train_end:], labels[train_end:],
                                          is_training=False)

print("\nProcessed images")
print("Train:", train_images.shape)
print("Valid:", valid_images.shape)

X_train, y_train = create_sequences(train_images, train_labels, SEQUENCE_LENGTH)
X_valid, y_valid = create_sequences(valid_images, valid_labels, SEQUENCE_LENGTH)


def build_model(rnn: str = "LAN"):
    """Shared conv trunk + the recurrent cell selected by ``rnn``.

    The five Conv2D blocks shrink 75x75x3 to a 24-vector per timestep; everything
    after GlobalAveragePooling2D is the part that differs between the cells under
    comparison.
    """
    inp = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    # Normalise to [-1, 1] inside the graph so the test node can feed raw camera
    # frames straight to the saved model.
    x = TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0))(inp)
    x = TimeDistributed(Conv2D(64, (7, 7), strides=(2, 2), activation="relu",
                               padding="same", kernel_regularizer=l2(1e-5)))(x)
    x = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation="relu",
                               padding="same", kernel_regularizer=l2(1e-5)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(3, 3), activation="relu",
                               padding="same", kernel_regularizer=l2(3.02e-5)))(x)
    x = TimeDistributed(Conv2D(48, (7, 7), strides=(3, 3), activation="relu",
                               padding="same", kernel_regularizer=l2(1e-5)))(x)
    x = TimeDistributed(Conv2D(24, (7, 7), strides=(2, 2), activation="relu",
                               padding="same", kernel_regularizer=l2(1.4e-5)))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    if rnn == "NAC":
        x = NAC(d_model=64, num_heads=16, sparsity=0.8, topk=16, activation="tanh",
                return_sequences=False, use_riemann_sum=False)(x)
    elif rnn == "LSTM":
        x = LSTM(64, activation="tanh", return_sequences=False)(x)
    elif rnn == "LAN":
        x = LAN(d_model=64, num_heads=16, topk=16, activation="tanh",
                return_sequences=False)(x)
    elif rnn == "SDPA":
        x = MultiHeadAttention(key_dim=64, num_heads=16)(x, x)
        x = Activation("tanh")(x)
        x = Flatten()(x)
    elif rnn == "LTC":
        x = Dense(32, activation="relu")(x)
        return Model(inputs=inp, outputs=LTC(AutoNCP(units=32, output_size=2),
                                             return_sequences=False)(x))
    elif rnn == "CfC":
        x = Dense(32, activation="relu")(x)
        return Model(inputs=inp, outputs=CfC(AutoNCP(units=32, output_size=2),
                                             return_sequences=False)(x))
    else:
        raise ValueError(f"Unknown rnn policy: {rnn!r}")

    x = Dense(32, activation="relu")(x)
    out = Dense(2, activation="linear")(x)
    return Model(inputs=inp, outputs=out)


model = build_model(TRAIN_POLICY)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
model.summary()

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
# Two checkpoints: the rolling best-weights file (what the test node loads) and
# the final state after training stops.
best_weights_path = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_{TRAIN_POLICY}_best.weights.h5")
final_model_path = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_{TRAIN_POLICY}.keras")

callbacks = [
    # Halve-and-slow LR decay: 0.9^epoch, so a plateau costs a little progress
    # rather than a restart.
    ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=1, min_lr=1e-10),
    ModelCheckpoint(best_weights_path, monitor="val_loss", save_best_only=True,
                    mode="min"),
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
]

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks,
    verbose=1,
)
model.save(final_model_path)

results = model.evaluate(X_valid, y_valid, batch_size=BATCH_SIZE, verbose=1)
print(f"Test loss: {results[0]:.6f}")
print(f"Test MAE : {results[1]:.6f}")
