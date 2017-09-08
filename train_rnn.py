""" Actually train the RNN """
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard

model = Sequential()
model.add(LSTM(700, input_shape=(None, 12+12+49), return_sequences=True))
model.add(LSTM(700, return_sequences=True))
model.add(TimeDistributed(Dense(49)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

from prepare_data import seq_generator
generator = seq_generator()

callbacks = []
callbacks.append(ModelCheckpoint('2songs.h5', monitor='loss'))
callbacks.append(TensorBoard())
model.fit_generator(generator, 10, epochs=1000, callbacks=callbacks)
