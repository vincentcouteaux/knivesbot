""" The goal of the modul is to have data ready to feed the RNN.
The input contains the bass, stacked with the chord, stacked with the PREVIOUS
NOTE. the output contains the CURRENT NOTE """
import glob
import numpy as np
from parsemidi import get_input, sync_with_all_transp, sync_all_transp_bass_chords
FILES = [name[:-9] for name in glob.glob('mididb/*bass.mid')]
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

def get_all_database_bidirectional():
    """ the same function than above for the bidirectional setting:
    only bass and chords are input of the network """
    inputs_seq = []
    outputs_seq = []
    length_of_each_seq = []
    for filename in FILES:
        print(filename)
        bass_roll, chords_roll, solo_roll = get_input(filename)
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

def seq_generator(bidirectional=False):
    if not bidirectional:
        inputs_seq, outputs_seq, lengths = get_all_database()
    else:
        inputs_seq, outputs_seq, lengths = get_all_database_bidirectional()
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

if __name__ == "__main__":
    gene = seq_generator()
    inputs_seq, outputs_seq, lengths = get_all_database()
    proba = np.array(lengths).astype(float)
    proba /= np.sum(proba)
    print(proba)
    print(draw_distrib(proba))
    print(draw_distrib(proba))
    print(draw_distrib(proba))
    print(draw_distrib(proba))
    print(draw_distrib(proba))
