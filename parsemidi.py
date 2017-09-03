""" This module transform the midi file into inputs to the neural network """
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

def quantify2eights(sheet):
    """ quantify to 1/8th a sheet as output by absolute_note_on """
    out = np.zeros(int(round(2*sheet[-1].time))+1) - 1
    for event in sheet:
        out[int(round(2*event.time))] = event.note
    return out

def melody_midi2_1hot(midi_file):
    """ return a piano roll-like numpy array """
    eights = quantify2eights(absolute_note_on(midi_file)).astype(int)
    out = np.zeros((eights.shape[0], 49)).astype(int) #48 is for octave + 1 for no note
    for i, note in enumerate(eights):
        if note == -1:
            out[i, 48] = 1
        else:
            out[i, (note - 48)%49] = 1
    return out

def chords2_1hot(midi_file):
    """ return the piano roll of the bass or the chord.
        we don't care about the octave, so we return a
        (len(midi_file), 12) sized np array. Also, no rest, a note is held until
        a new one appears """
    sheet = absolute_note_on(midi_file)
    out = np.zeros((int(round(2*sheet[-1].time))+1, 12))
    for event in sheet:
        out[int(round(2*event.time)), event.note%12] = 1.
    prev_index = []
    for pianoroll_slice in out:
        if pianoroll_slice.any():
            prev_index = np.where(pianoroll_slice)[0]
        else:
            pianoroll_slice[prev_index] = 1.
    return out

def sync_bass_chords_solo(roll_bass, roll_chords, roll_solo):
    """ TODO concatenate the 3 rolls into 1. Starts from the first note of the bass,
    to the last note of the solo. continue the last chord if the solo last longer """
    first_non_void = np.argmax(roll_bass.any(1))
    bass_padded = np.pad(roll_bass[first_non_void:],
                         ((0, roll_solo.shape[0] - roll_bass.shape[0]), (0, 0)), 'edge')
    chords_padded = np.pad(roll_chords[first_non_void:],
                           ((0, roll_solo.shape[0] - roll_chords.shape[0]), (0, 0)), 'edge')
    return np.concatenate((bass_padded, chords_padded, roll_solo[first_non_void:]), 1)

def get_input(name):
    """! """
    bass_file = mido.MidiFile(name + '_bass.mid')
    chords_file = mido.MidiFile(name + '_chords.mid')
    solo_file = mido.MidiFile(name + '_solo.mid')
    roll_solo = melody_midi2_1hot(solo_file)
    roll_bass = chords2_1hot(bass_file)
    roll_chords = chords2_1hot(chords_file)
    return sync_bass_chords_solo(roll_bass, roll_chords, roll_solo)


def get_bpm(midi_file):
    """! returns the tempo in beat per minute, of a midi file """
    return 60000000/get_tempo(midi_file)

def get_tempo(midi_file):
    """! returns the tempo in microsecond per beat """
    for msg in midi_file:
        if msg.type == 'set_tempo':
            return msg.tempo

NOTE2STR = {0: 'C', 1:'C#', 2:'D', 3:'Eb', 4:'E', 5:'F', 6:'F#',
            7:'G', 8:'G#', 9:'A', 10:'Bb', 11:'B'}
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
        X = get_input('mididb/youdBeSo')
        for x in X:
            print x
    sync()
