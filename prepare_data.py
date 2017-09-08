""" The goal of the modul is to have data ready to feed the RNN.
The input contains the bass, stacked with the chord, stacked with the PREVIOUS
NOTE. the output contains the CURRENT NOTE """
import glob
import numpy as np
from parsemidi import get_input, sync_with_all_transp
FILES = [name[:-9] for name in glob.glob('mididb/*bass.mid')]

def get_all_database():
    inputs_seq = []
    outputs_seq = []
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
        print(inputs_seq[i].shape, outputs_seq[i].shape)
    return inputs_seq, outputs_seq

def seq_generator():
    inputs_seq, outputs_seq = get_all_database()
    n_seq = len(inputs_seq)
    while True:
        index = np.random.randint(n_seq)
        yield np.expand_dims(inputs_seq[index], 0), \
              np.expand_dims(outputs_seq[index], 0)

if __name__ == "__main__":
    gene = seq_generator()
    print(next(gene))
    print(next(gene))
    print(next(gene))
