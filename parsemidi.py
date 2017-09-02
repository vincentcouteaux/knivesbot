""" This module transform the midi file into inputs to the neural network """
import mido
#import numpy as np

def absolute_note_on(midi_file):
    """ return a list of midi onset event """
    abs_time = 0.
    class OnsetEvent:
        """ struct to encapsulate the time and note of a event """
        def __init__(self, time, note):
            self.time = time
            self.note = note
    sheet = []
    for msg in midi_file:
        if msg.type == 'note_on':
            sheet.append(OnsetEvent(abs_time, msg.note))
        abs_time += msg.time
    return sheet

def get_BPM(midi_file):
    for msg in midi_file:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            return 60000000/tempo

if __name__ == "__main__":
    def main():
        mid = mido.MidiFile('mididb/allTheThings_bass.mid')
        sheet = absolute_note_on(mid)
        for event in sheet:
            print("{}: {}".format(event.time, event.note))
        print(get_BPM(mid))
        for msg in mid:
            if msg.is_meta:
                print(msg)
                print(msg.type)
    main()
