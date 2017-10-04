""" The goal of the modul is to have data ready to feed the RNN.
The input contains the bass, stacked with the chord, stacked with the PREVIOUS
NOTE. the output contains the CURRENT NOTE """
import glob
import numpy as np
from parsemidi import get_input, sync_with_all_transp, sync_all_transp_bass_chords
WALKING_BASS = True # True if we learn the walking bass, False if we learn the solo
if not WALKING_BASS:
    FILES = [name[:-9] for name in glob.glob('mididb/*bass.mid')]
else:
    FILES = [name[:-12] for name in glob.glob('mididb/*walking.mid')]
print(FILES)
SEQ_LENGTH = 16*8 # 16 bars of 8 eighth notes

def get_all_database():
    inputs_seq = []
    outputs_seq = []
    length_of_each_seq = []
    for filename in FILES:
        print(filename)
        bass_roll, chords_roll, solo_roll = get_input(filename)
        inp, out = sync_with_all_transp(bass_roll, chords_roll, solo_roll)
        for x in inp:
            print(x.shape)
        for x in out:
            print(x.shape)
        inputs_seq += inp
        outputs_seq += out
    for i in range(len(inputs_seq)):
        length_of_each_seq.append(inputs_seq[i].shape[0])
    return inputs_seq, outputs_seq, length_of_each_seq

def get_all_database_bidirectional(walking_bass=False):
    """ the same function than above for the bidirectional setting:
    only bass and chords are input of the network """
    inputs_seq = []
    outputs_seq = []
    length_of_each_seq = []
    for filename in FILES:
        print(filename)
        bass_roll, chords_roll, solo_roll = get_input(filename, 
                                                      4 if walking_bass else 8,
                                                      walking_bass)
        inp, out = sync_all_transp_bass_chords(bass_roll, chords_roll, solo_roll)
        for x in inp:
            print(x.shape)
        for x in out:
            print(x.shape)
        inputs_seq += inp
        outputs_seq += out
    for i in range(len(inputs_seq)):
        length_of_each_seq.append(inputs_seq[i].shape[0])
    return inputs_seq, outputs_seq, length_of_each_seq

def draw_distrib(distrib):
    cumsum = np.cumsum(distrib)
    r = np.random.rand()
    return np.argmax(cumsum > r)

def seq_generator(bidirectional=False, walking_bass=False):
    if not bidirectional:
        inputs_seq, outputs_seq, lengths = get_all_database()
    else:
        inputs_seq, outputs_seq, lengths = \
            get_all_database_bidirectional(walking_bass)
    proba = np.array(lengths).astype(float)
    proba /= np.sum(proba)
    n_seq = len(inputs_seq)
    while True:
        #index = np.random.randint(n_seq) # a random midi file
        index = draw_distrib(proba)
        start = np.random.randint(outputs_seq[index].shape[0] - SEQ_LENGTH)
        start = start if start > 0 else 0
        yield np.expand_dims(inputs_seq[index][start:start+SEQ_LENGTH], 0), \
              np.expand_dims(outputs_seq[index][start:start+SEQ_LENGTH], 0)

def seq_multidir(walking_bass=False):
    bi_gen = seq_generator(True, walking_bass)
    while True:
        inp, out = next(bi_gen)
        delayed_solo = np.roll(out, 1, 1)
        delayed_solo[0, 0, :] = 0.
        delayed_solo[0, 0, -1] = 1.
        yield [inp, delayed_solo], out
    

if __name__ == "__main__":
    from parsemidi import midi_note2str
    gene = seq_multidir(True)
    in_, out = next(gene)
    chords = in_[0]
    d_solo = in_[1]
    print('Chords:')
    print(map(midi_note2str, np.argmax(chords[0][:, :12], 1)))
    #for chord in chords[0]:
    #    print(chord)
    print('solo:')
    print(map(midi_note2str, np.argmax(out[0], 1)%12))
    print('delayed solo:')
    print(np.argmax(d_solo[0], 1))

