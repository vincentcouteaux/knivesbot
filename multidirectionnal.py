""" train the RNN with a particular structure. The chords are passed through a
bidirectional LSTM layer, while the solo notes are passed through a simple LSTM layer.
"""
from keras.models import Model
from keras.layers import LSTM, TimeDistributed, Activation, Input, Bidirectional
from keras.layers import Dense, Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
from prepare_data import seq_multidir

WALKING_BASS = True
chords_tens = Input(shape=(None, 24), name='chords_input')
solo_tens = Input(shape=(None, 37 if WALKING_BASS else 49), name='solo_input') #the solo must be delayed by 1 eigth so that
                    # it contains the previously played note

blstm_chords = Bidirectional(LSTM(500, return_sequences=True), name="chords_blstm")(chords_tens)
#blstm_chords = Bidirectional(LSTM(500, return_sequences=True),
#                               name="chords_blstm")(blstm_chords)
lstm_solo = LSTM(500, return_sequences=True)(solo_tens)
#lstm_solo = LSTM(500, return_sequences=True)(lstm_solo)

#concat = K.concatenate((blstm_chords, lstm_solo), 1)
concat = Concatenate()([blstm_chords, lstm_solo])
#lstm_concat = LSTM(500, return_sequences=True)(concat)
time_dis = TimeDistributed(Dense(49 if not WALKING_BASS else 37))(concat)
softmax = Activation('softmax')(time_dis)
model = Model(inputs=[chords_tens, solo_tens], outputs=softmax)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

callbacks = []
callbacks.append(ModelCheckpoint('shallow_walking_good.h5', monitor='loss'))
callbacks.append(TensorBoard())

generator = seq_multidir(WALKING_BASS)
a, b = next(generator)
print('"""""')
print(a[0].shape, a[1].shape)
print(b.shape)
model.fit_generator(generator, 50, epochs=1000, callbacks=callbacks)
