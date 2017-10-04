""" this module is used to generate a melody from the complex "multidirectional" model, where the chords is passed through a bidirectional layer, and the solo an 
normal LSTM. """
import numpy as np
import keras.backend as K

def get_functions(model):
    """! return keras functions to evaluate tensors of a trained models. works for simple multidirectional models, without names """
    blstm_chords = model.layers[2].get_output_at(0) # in this model the second layer
                                                    # is the blstm
    chords_input = model.layers[2].get_input_at(0)
    solo_input = model.layers[3].get_input_at(0)
    get_chords_features = K.function([chords_input], [blstm_chords])
    get_solo = K.function([solo_input, blstm_chords], model.outputs)
    return get_chords_features, get_solo

def gen_melody(model, roll_bass, roll_chords, context=16):
    basschords_roll = np.concatenate((roll_bass, roll_chords), 1)
    get_chords_features, get_solo = get_functions(model)
    chords_features = get_chords_features([np.expand_dims(basschords_roll, 0)])[0]
    melody = []
    solo_roll = np.zeros((1, roll_bass.shape[0], 49))
    solo_roll[0, 0, np.random.randint(49)] = 1.
    for i in range(roll_bass.shape[0] - 1):
        beginning = i-context if i - context > 0 else 0
        melody.append(np.argmax(get_solo([solo_roll[:, beginning:i+1, :], chords_features[:, beginning:i+1, :]])[0][0], 1)[-1])
        solo_roll[:, i+1, melody[-1]] = 1.
    return melody

if __name__ == "__main__":
    from generate_melody import print_melody, test_chords, TEST_BASS, TEST_CHORDS
    from playback import play_melody_and_chords
    from keras.models import load_model
    model = load_model("models/multidirectional.h5")
    rolls = test_chords()
    for song in rolls:
        print("")
        print_melody(gen_melody(model, song[0], song[1]))
        print("")
    melody = gen_melody(model, TEST_BASS, TEST_CHORDS)
    print_melody(melody)
    play_melody_and_chords(melody, TEST_BASS, TEST_CHORDS, 190, True)
