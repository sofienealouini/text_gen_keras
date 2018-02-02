import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from data import NB_CHARS, INPUTS, TARGETS
from params import NB_ITER, BATCH_SIZE, SEQ_LEN
import argparse


# Arguments à fournir dans le terminal
parser = argparse.ArgumentParser(description='trains the model')
parser.add_argument('-m','--model',help='model to start with')
parser.add_argument('--gpu', help='using gpu', action='store_true')
args = parser.parse_args()


# Fixer le nombre de threads utilisés par Keras (si CPU)
if not args.gpu:
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True)
    session = tf.Session(config=config)
    K.set_session(session)


# Définition de checkpoints
filepath = "/output/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# Modèle
if args.model is None:
    model = Sequential()
    model.add(LSTM(256, input_shape=(SEQ_LEN, NB_CHARS), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(NB_CHARS))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
else:
    model = load_model(args.model)

# Commencer le training
model.fit(INPUTS, TARGETS, batch_size=BATCH_SIZE, epochs=NB_ITER, callbacks=callbacks_list, verbose=1)