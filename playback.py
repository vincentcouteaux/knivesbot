"""! This module is used to play a generated melody. On OSX you must have a synth
running, like SimpleSynth """
import time
import numpy as np
import pygame.midi

pygame.midi.init()
#PLAYER = pygame.midi.Output(pygame.midi.get_default_output_id())
PLAYER = pygame.midi.Output(0)
PLAYER.set_instrument(2)

def play_note(midi_note, tempo, swing):
    """ @param swing is -1 if you want no swing, 0 if its the first 8th,
    1 if its the second (delayed) 8th
    @param tempo is in BPM """
    if midi_note != -1:
        PLAYER.note_on(midi_note, 127, 4)
    if swing == -1:
        to_sleep = 30./tempo
    elif swing == 0:
        to_sleep = 34./tempo
    else:
        to_sleep = 26./tempo
    time.sleep(to_sleep)
    if midi_note != -1:
        PLAYER.note_off(midi_note, 127, 4)

def play_melody(melody, tempo, swing):
    """ swing is a boolean """
    for i, note in enumerate(melody):
        to_play = -1
        if note <= 47:
            to_play = note + 48
        play_note(to_play, tempo, i%2 if swing else -1)

def play_melody_and_chords(melody, roll_bass, roll_chords, tempo, swing):
    """! """
    chords_note = np.array([-1, -1])
    bass_note = -1
    melody_note = 48
    for i, _ in enumerate(melody):
        #print(i, melody[i])
        if melody_note != 48:
            PLAYER.note_off(melody_note+48, 70, 4)
        melody_note = melody[i]
        if melody_note != 48:
            PLAYER.note_on(melody_note+48, 70, 4)
            #print(melody_note)
        current_bass = np.argmax(roll_bass[i])
        if current_bass != bass_note or i%8 == 0:
            PLAYER.note_off(bass_note + 48, 60, 2)
            bass_note = current_bass
            #print("bar #{}, BASS {}".format(i, bass_note))
            PLAYER.note_on(current_bass + 48, 60, 2)
        current_chords = np.where(roll_chords[i])[0]
        if (current_chords != chords_note).any() or i%8 == 0:
            PLAYER.note_off(chords_note[0]+60, 60, 2)
            PLAYER.note_off(chords_note[1]+60, 60, 2)
            chords_note = current_chords
            #print("bar #{}, CHORDS {}, {}".format(i, chords_note[0], chords_note[1]))
            PLAYER.note_on(chords_note[0]+60, 60, 2)
            PLAYER.note_on(chords_note[1]+60, 60, 2)
        if swing == False:
            to_sleep = 30./tempo
        elif i%2 == 0:
            to_sleep = 34./tempo
        else:
            to_sleep = 26./tempo
        time.sleep(to_sleep)

