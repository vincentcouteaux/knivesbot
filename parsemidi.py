""" This module transform the midi file into inputs to the neural network """
from __future__ import print_function
import mido
import numpy as np

#def absolute_from_ticks(midi_name, ticks_per_beat):
#    track = midi.read_midifile(midi_name)[0]
#    tick = 0
#    for event in track:
#        tick += event.tick
#        if isinstance(event, midi.NoteOnEvent):
#            print float(tick)/ticks_per_beat, midi_note2str(event.data[0]), event.tick
#        else:
#            print(event.tick)

def absolute_note_on(midi_file):
    """ return a list of midi onset event """
    abs_time = 0.
    tempo = float(get_tempo(midi_file))/1000000
    class OnsetEvent(object):
        """ struct to encapsulate the time and note of a event """
        def __init__(self, time, note):
            self.time = time
            self.note = note
    sheet = []
    for msg in midi_file:
#        print(msg)
        abs_time += msg.time
        if msg.type == 'note_on':
            sheet.append(OnsetEvent((abs_time)/tempo, msg.note))
    return sheet

def quantify(sheet, rythm):
    """ quantify a sheet as output by absolute_note_on.
    @param rythm (1, 2, 4, 8, 16...) 1/rythm at witch to quantify"""
    out = np.zeros(int(round((rythm/4.)*sheet[-1].time))+1) - 1
    for event in sheet:
        out[int(round((rythm/4.)*event.time))] = event.note
    return out

def quantify2eights(sheet):
    """ quantify to 1/8th a sheet as output by absolute_note_on """
    return quantify(sheet, 8)

def quantify2quarters(sheet):
    """ quantify to 1/8th a sheet as output by absolute_note_on """
    return quantify(sheet, 4)

def melody_midi2_1hot(midi_file, quant=8):
    """ return a piano roll-like numpy array """
    eights = quantify(absolute_note_on(midi_file), quant).astype(int)
    out = np.zeros((eights.shape[0], 49)).astype(int) #48 is four octave + 1 for no note
    for i, note in enumerate(eights):
        if note == -1:
            out[i, 48] = 1
        else:
            #out[i, (note - 48)%49] = 1
            out[i, (note - 48)%48] = 1 # 48 makes more sense right ?
    return out

def walking_midi2_1hot(midi_file, quant=4):
    """ same, but in the bass range """
    quarters = quantify(absolute_note_on(midi_file), quant).astype(int)
    out = np.zeros((quarters.shape[0], 37)).astype(int) #37 is four octave + 1 for no note
    for i, note in enumerate(quarters):
        #print(note)
        if note == -1:
            out[i, 36] = 1
        else:
            out[i, (note - 36)%36] = 1 #not sure if 24 or 12
    return out

def chords2_1hot(midi_file, quant=8):
    """ return the piano roll of the bass or the chord.
        we don't care about the octave, so we return a
        (len(midi_file), 12) sized np array. Also, no rest, a note is held until
        a new one appears """
    sheet = absolute_note_on(midi_file)
    out = np.zeros((int(round((quant/4.)*sheet[-1].time))+1, 12))
    for event in sheet:
        out[int(round((quant/4.)*event.time)), event.note%12] = 1.
    prev_index = []
    for pianoroll_slice in out:
        if pianoroll_slice.any():
            prev_index = np.where(pianoroll_slice)[0]
        else:
            pianoroll_slice[prev_index] = 1.
    return out

def trim(roll_bass, roll_chords, roll_solo):
    """ sometimes the midi file does not start from zeros, so we trim
    what comes before, and we continue the last chord for the bass and chords
    until the last solo note """
    first_non_void = np.argmax(roll_bass.any(1))
    bass_padded = np.pad(roll_bass[first_non_void:],
                         ((0, roll_solo.shape[0] - roll_bass.shape[0]), (0, 0)), 'edge')
    chords_padded = np.pad(roll_chords[first_non_void:],
                           ((0, roll_solo.shape[0] - roll_chords.shape[0]), (0, 0)), 'edge')
    return bass_padded, chords_padded, roll_solo[first_non_void:]

def sync_bass_chords_solo(roll_bass, roll_chords, roll_solo):
    r""" Concatenate the 3 rolls into 1. The piano rolls must already be trimmed.
    /!\ /!\ The solo is delayed by one box since the input contains the previous
    note played /!\ /!\ """
    #first_non_void = np.argmax(roll_bass.any(1))
    #bass_padded = np.pad(roll_bass[first_non_void:],
    #                     ((0, roll_solo.shape[0] - roll_bass.shape[0]), (0, 0)), 'edge')
    #chords_padded = np.pad(roll_chords[first_non_void:],
    #                       ((0, roll_solo.shape[0] - roll_chords.shape[0]), (0, 0)), 'edge')
    #if first_non_void > 0:
    #    return np.concatenate((bass_padded, chords_padded, roll_solo[first_non_void-1:-1]), 1)
    blank_note = np.zeros((1, 49))
    blank_note[0, -1] = 1.
    return np.concatenate((roll_bass, roll_chords, 
                        np.concatenate((blank_note, roll_solo[:-1]))), 1)

def get_input(name, quant=8, walking_bass=False):
    """! """
    bass_file = mido.MidiFile(name + '_bass.mid')
    chords_file = mido.MidiFile(name + '_chords.mid')
    suffix = '_walking.mid' if walking_bass else '_solo.mid'
    solo_file = mido.MidiFile(name + suffix)
    if not walking_bass:
        roll_solo = melody_midi2_1hot(solo_file, quant)
    else:
        roll_solo = walking_midi2_1hot(solo_file, quant)
    roll_bass = chords2_1hot(bass_file, quant)
    roll_chords = chords2_1hot(chords_file, quant)
    return roll_bass, roll_chords, roll_solo
    #return sync_bass_chords_solo(roll_bass, roll_chords, roll_solo)

def all_transpositions(rolls, solo=False):
    """! return a list of all the transpositions of the piano rolls.
    Solo piano rolls are special because there last slot is 1 if no note is played, 
    therefore must not be transpose"""
    out = [np.zeros(rolls.shape) for i in range(12)]
    for i, slice_ in enumerate(rolls):
        for transp in range(12):
            if solo:
                out[transp][i, :-1] = np.roll(slice_[:-1], transp)
                out[transp][i, -1] = slice_[-1]
            else:
                out[transp][i] = np.roll(slice_, transp)
    #out.append(rolls)
    return out

def sync_with_all_transp(roll_bass, roll_chords, roll_solo):
    """ apply all_transpositions to the 2 rolls, and concatenate them with
    sync_bass_chords_solo """
    roll_bass, roll_chords, roll_solo = trim(roll_bass, roll_chords, roll_solo)
    all_bass = all_transpositions(roll_bass, False)
    all_chords = all_transpositions(roll_chords, False)
    all_solo = all_transpositions(roll_solo, True)
    all_sync = []
    for i in range(12):
        all_sync.append(sync_bass_chords_solo(all_bass[i], all_chords[i], all_solo[i]))
    return all_sync, all_solo

def sync_all_transp_bass_chords(roll_bass, roll_chords, roll_solo):
    """ concatenate only the bass and the chords, and does not delay the solo.
    This is the the training setting where only the bass and chords are input
    of the network """
    roll_bass, roll_chords, roll_solo = trim(roll_bass, roll_chords, roll_solo)
    all_bass = all_transpositions(roll_bass, False)
    all_chords = all_transpositions(roll_chords, False)
    all_solo = all_transpositions(roll_solo, True)
    all_sync = []
    for i in range(12):
        all_sync.append(np.concatenate((all_bass[i], all_chords[i]), 1))
    return all_sync, all_solo

def get_bpm(midi_file):
    """! returns the tempo in beat per minute, of a midi file """
    return 60000000/get_tempo(midi_file)

def get_tempo(midi_file):
    """! returns the tempo in microsecond per beat """
    for msg in midi_file:
        if msg.type == 'set_tempo':
            return msg.tempo

NOTE2STR = {0: 'C ', 1:'C#', 2:'D ', 3:'Eb', 4:'E ', 5:'F ', 6:'F#',
            7:'G ', 8:'G#', 9:'A ', 10:'Bb', 11:'B '}
def midi_note2str(midi_note):
    """ return the name of the midi note """
    if midi_note < 0:
        return '-'
    return NOTE2STR[midi_note%12]


if __name__ == "__main__":
    def main():
        mid = mido.MidiFile('mididb/youdBeSo_solo.mid')
        sheet = absolute_note_on(mid)
        print(map(midi_note2str, quantify2eights(sheet)))
        piano_roll = melody_midi2_1hot(mid)
        for one_hot in piano_roll:
            print(np.array_repr(one_hot).replace('\n', ''))
        print(get_bpm(mid))
        for msg in mid:
            if msg.is_meta:
                print(msg)
        print(mid.ticks_per_beat)
    def chords():
        mid = mido.MidiFile('mididb/youdBeSo_bass.mid')
        piano_roll = chords2_1hot(mid)
        for slice_ in  piano_roll:
            print(slice_)
    def sync():
        X_bass, X_chords, X_solo = get_input('mididb/allTheThings')
        print(X_solo.shape, X_bass.shape)
        for x in X_chords:
            print(x)
        #X_solo[-2] = 0
        #X_solo[-2, -1] = 1
        all_transp_X, all_y = sync_with_all_transp(X_bass, X_chords, X_solo)
        for i, transp in enumerate(all_transp_X):
            print("Input: ")
            print(transp)
            print("Output: ")
            print(all_y[i].shape)
            print(all_y[i])
            print("\n")
    def walking():
        roll_bass, roll_chords, roll_walk = get_input('mididb/greenDolphin', 4, True)
        print(np.argmax(roll_walk[:20], 1))
        pass
    walking()
