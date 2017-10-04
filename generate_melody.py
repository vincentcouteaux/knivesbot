"""! This module generates a melody with a trained model """
from __future__ import print_function
from parsemidi import *
from keras.models import load_model
from keras.callbacks import Callback
from numpy.random import randint

def generate_melody(model, roll_bass, roll_chords, context=32):
    big_roll = np.concatenate((roll_bass, roll_chords, np.zeros((roll_bass.shape[0], 49))), 1)
    big_roll[0, randint(24, big_roll.shape[1])] = 1.
    big_roll = np.expand_dims(big_roll, 0)
    melody = []
    for i in range(big_roll.shape[1]-1):
        beginning = i-context if i-context > 0 else 0
        melody.append(np.argmax(model.predict(big_roll[:, beginning:i+1, :])[0], 1)[-1])
        big_roll[:, i+1, melody[-1]+24] = 1.
    return melody

def gen_melody_bidirectional(model, roll_bass, roll_chords):
    solo_roll = model.predict(np.expand_dims(np.concatenate((roll_bass, roll_chords), 1), 0))[0]
    return np.argmax(solo_roll, 1)

def print_melody(melody):
    names = map(midi_note2str, map(lambda x: x if x <= 47 else -1, melody))
    for i, name in enumerate(names):
        print(name+" ", end="")
        if (i+1)%8 == 0:
            print("| ", end="")
        if (i+1)%32 == 0:
            print("")
        if (i+1)%128 == 0:
            print("")

def test_chords():
    """ return a bass chords piano-rolls with: 2 bars of Cmaj, 2 bars of F#maj,
    1 ii-V-I in Gmaj, 1 ii-V-I in C#maj """
    bass_Cmaj = np.zeros((16, 12))
    bass_Cmaj[:, 0] = 1.
    chords_Cmaj = np.zeros((16, 12))
    chords_Cmaj[:, 4] = 1.
    chords_Cmaj[:, 11] = 1.
    bass_Fsmaj = np.zeros((16, 12))
    bass_Fsmaj[:, 6] = 1.
    chords_Fsmaj = np.zeros((16, 12))
    chords_Fsmaj[:, 10] = 1.
    chords_Fsmaj[:, 5] = 1.
    bass_iiVI_Gmaj = np.zeros((32, 12))
    bass_iiVI_Gmaj[:8, 9] = 1.
    bass_iiVI_Gmaj[8:16, 2] = 1.
    bass_iiVI_Gmaj[16:, 7] = 1.
    chords_iiVI_Gmaj = np.zeros((32, 12))
    chords_iiVI_Gmaj[:8, 0] = 1.
    chords_iiVI_Gmaj[:8, 7] = 1.
    chords_iiVI_Gmaj[8:16, 6] = 1.
    chords_iiVI_Gmaj[8:16, 0] = 1.
    chords_iiVI_Gmaj[16:, 11] = 1.
    chords_iiVI_Gmaj[16:, 6] = 1.
    bass_iiVI_Abmaj = np.zeros((32, 12))
    bass_iiVI_Abmaj[:8, 10] = 1.
    bass_iiVI_Abmaj[8:16, 3] = 1.
    bass_iiVI_Abmaj[16:, 8] = 1.
    chords_iiVI_Abmaj = np.zeros((32, 12))
    chords_iiVI_Abmaj[:8, 1] = 1.
    chords_iiVI_Abmaj[:8, 8] = 1.
    chords_iiVI_Abmaj[8:16, 7] = 1.
    chords_iiVI_Abmaj[8:16, 1] = 1.
    chords_iiVI_Abmaj[16:, 0] = 1.
    chords_iiVI_Abmaj[16:, 7] = 1.
    return [[bass_Cmaj, chords_Cmaj], [bass_Fsmaj, chords_Fsmaj],
            [bass_iiVI_Gmaj, chords_iiVI_Gmaj], [bass_iiVI_Abmaj, chords_iiVI_Abmaj]]

#TEST_BASS, TEST_CHORDS, _ = trim(*get_input('mididb/test_db/giantSteps'))
TEST_BASS, TEST_CHORDS, _ = trim(*get_input('mididb/test_db/bebopBlues'))
#TEST_BASS, TEST_CHORDS, _ = trim(*get_input('mididb/test_db/takeTheATrain'))
#TEST_BASS, TEST_CHORDS, _ = trim(*get_input('mididb/allTheThings'))
#TEST_BASS, TEST_CHORDS, _ = trim(*get_input('mididb/youdBeSo'))
ROLLS = test_chords()
class GenMelody(Callback):
    def on_epoch_end(self, epoch, logs={}):
        #print_melody(generate_melody(self.model, TEST_BASS, TEST_CHORDS))
        print("Cmaj7")
        print_melody(generate_melody(self.model, ROLLS[0][0], ROLLS[0][1]))
        print("")
        print("F#maj7")
        print_melody(generate_melody(self.model, ROLLS[1][0], ROLLS[1][1]))
        print("")
        print("Gmaj ii-V-I")
        print_melody(generate_melody(self.model, ROLLS[2][0], ROLLS[2][1]))
        print("")
        print("Abmaj ii-V-I")
        print_melody(generate_melody(self.model, ROLLS[3][0], ROLLS[3][1]))
        print("")

class GenMelodyBidirectional(Callback):
    def on_epoch_end(self, epoch, logs={}):
        #print_melody(generate_melody(self.model, TEST_BASS, TEST_CHORDS))
        print("Cmaj7")
        print_melody(gen_melody_bidirectional(self.model, ROLLS[0][0], ROLLS[0][1]))
        print("")
        print("F#maj7")
        print_melody(gen_melody_bidirectional(self.model, ROLLS[1][0], ROLLS[1][1]))
        print("")
        print("Gmaj ii-V-I")
        print_melody(gen_melody_bidirectional(self.model, ROLLS[2][0], ROLLS[2][1]))
        print("")
        print("Abmaj ii-V-I")
        print_melody(gen_melody_bidirectional(self.model, ROLLS[3][0], ROLLS[3][1]))
        print("")

if __name__ == "__main__":
    from playback import play_melody_and_chords, play_melody
    model = load_model('models/8songs.h5')
    rolls = test_chords()
    for song in rolls:
        print("")
        print_melody(generate_melody(model, song[0], song[1]))
        print("")
    print("take the A train")
    length = 44*8
    melody = generate_melody(model, TEST_BASS[:length], TEST_CHORDS[:length], 16)
    print_melody(melody)
    play_melody_and_chords(melody, TEST_BASS, TEST_CHORDS, 220, True)
    #play_melody(melody, 170, True)
