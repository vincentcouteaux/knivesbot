""" Actually train the RNN """
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from generate_melody import GenMelodyBidirectional

model = Sequential()
model.add(Bidirectional(LSTM(500,  return_sequences=True), input_shape=(None, 12+12)))
model.add(Bidirectional(LSTM(500, return_sequences=True)))
model.add(TimeDistributed(Dense(49)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

from prepare_data import seq_generator
generator = seq_generator(True)

callbacks = []
callbacks.append(ModelCheckpoint('bidirectional.h5', monitor='loss'))
callbacks.append(TensorBoard())

callbacks.append(GenMelodyBidirectional())
model.fit_generator(generator, 50, epochs=1000, callbacks=callbacks)
