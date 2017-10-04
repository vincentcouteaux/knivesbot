"""! like gen_melody_multi, but for walking basses """
import numpy as np
from gen_melody_multi import get_functions, gen_melody
from generate_melody import print_melody, test_chords, TEST_BASS, TEST_CHORDS
from keras.models import load_model
import keras.backend as K

def get_functions_from_names(model):
    """! works for models whose layers have names """
    blstm_chords = model.get_layer('chords_blstm').get_output_at(0) 
    chords_input = model.get_layer('chords_input').get_input_at(0)
    solo_input = model.get_layer('solo_input').get_input_at(0)
    get_chords_features = K.function([chords_input], [blstm_chords])
    get_solo = K.function([solo_input, blstm_chords], model.outputs)
    return get_chords_features, get_solo

def draw_distrib(distrib):
    cumsum = np.cumsum(distrib)
    r = np.random.rand()
    return np.argmax(cumsum > r)

def gen_walking(model, roll_bass, roll_chords, context=16, has_names=False):
    basschords_roll = np.concatenate((roll_bass, roll_chords), 1)
    if has_names:
        get_chords_features, get_solo = get_functions_from_names(model)
    else:
        get_chords_features, get_solo = get_functions(model)
    chords_features = get_chords_features([np.expand_dims(basschords_roll, 0)])[0]
    melody = []
    solo_roll = np.zeros((1, roll_bass.shape[0], 37))
    solo_roll[0, 0, 36] = 1.
    for i in range(roll_bass.shape[0] - 1):
        beginning = i-context if i - context > 0 else 0
        #melody.append(np.argmax(get_solo([solo_roll[:, beginning:i+1, :], chords_features[:, beginning:i+1, :]])[0][0], 1)[-1])
        melody.append(draw_distrib(get_solo([solo_roll[:, beginning:i+1, :], chords_features[:, beginning:i+1, :]])[0][0, -1]))
        solo_roll[:, i+1, melody[-1]] = 1.
    return melody
if __name__ == "__main__":
    from playback import play_melody_and_chords
    model = load_model("shallow_walking_good.h5")
    rolls = test_chords()
    for song in rolls:
        print("")
        print(song[0].shape, song[0][::2].shape)
        print_melody(gen_walking(model, song[0][::2], song[1][::2], has_names=True))
        print("")
    print("\n Actual grid")
    print(TEST_BASS.shape)
    melody = gen_walking(model, TEST_BASS[::2], TEST_CHORDS[::2])
    print_melody(melody)
    play_melody_and_chords(melody, TEST_BASS[::2], TEST_CHORDS[::2], 180,     
                           is_walking=True)
