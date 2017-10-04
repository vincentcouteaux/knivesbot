"""! This module is used to play a generated melody. On OSX you must have a synth
running, like SimpleSynth """
import time
import numpy as np
import pygame.midi

pygame.midi.init()
#PLAYER = pygame.midi.Output(pygame.midi.get_default_output_id())
PLAYER = pygame.midi.Output(0)
PLAYER.set_instrument(2)
SOLO_CHAN = 4
WALKING_CHAN = 3
SOLO_OFFSET = 48
WALKING_OFFSET = 24

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

def gen_velocities(melody):
    """ Generate the accentuation on a generated solo. Simple rule:
    accentuate all peaks in the melody. returns a vector with the velocities."""
    mel_copy = np.array(melody)
    mel_copy[mel_copy == 48] == 0
    velocities = np.zeros(len(melody), dtype=int) + 60
    for i, note in enumerate(mel_copy[:-1]):
        if note > mel_copy[i - 1] and note >= mel_copy[i+1]:
            velocities[i] = 100
    return velocities


def play_melody_and_chords(melody, roll_bass, roll_chords, tempo, swing=True, is_walking=False):
    """! if it's a walking bass, then we play quarters instead of eights,
    change the instrument to channel 3, and add only 3 octaves (36 midi notes)"""
    chords_note = np.array([-1, -1])
    bass_note = -1
    melody_note = 48
    to_add = 48 if not is_walking else 24
    channel = 4 if not is_walking else 3
    if not is_walking:
        velocities = gen_velocities(melody)
    for i, _ in enumerate(melody):
        #print(i, melody[i])
        if melody_note != 48:
            PLAYER.note_off(melody_note+to_add, 
                            60 if is_walking else velocities[i], channel)
        melody_note = melody[i]
        if melody_note != 48:
            PLAYER.note_on(melody_note+to_add,
                           60 if is_walking else velocities[i], channel)
            #print(melody_note)
        current_bass = np.argmax(roll_bass[i])
        if current_bass != bass_note or i%8 == 0:
            PLAYER.note_off(bass_note + 48, 60, 2)
            bass_note = current_bass
            #print("bar #{}, BASS {}".format(i, bass_note))
            PLAYER.note_on(current_bass + 48, 60, 2)
        current_chords = np.where(roll_chords[i])[0]
        if current_chords.size == 0:
            current_chords = np.array([-1, -1])
        if (current_chords != chords_note).any() or i%8 == 0:
            PLAYER.note_off(chords_note[0]+60, 60, 2)
            PLAYER.note_off(chords_note[1]+60, 60, 2)
            chords_note = current_chords
            #print("bar #{}, CHORDS {}, {}".format(i, chords_note[0], chords_note[1]))
            PLAYER.note_on(chords_note[0]+60, 60, 2)
            PLAYER.note_on(chords_note[1]+60, 60, 2)
        if is_walking:
            to_sleep = 60./tempo
        elif not swing:
            to_sleep = 30./tempo
        elif i%2 == 0:
            to_sleep = 34./tempo
        else:
            to_sleep = 26./tempo
        time.sleep(to_sleep)
    # off all notes, the very dirty way :(
    for note in range(128):
        PLAYER.note_off(note, channel=2)
        PLAYER.note_off(note, channel=channel)

def play_all(roll_bass, roll_chords, tempo, solo=None, walking=None, swing=True):
    """! if it's a walking bass, then we play quarters instead of eights,
    change the instrument to channel 3, and add only 3 octaves (36 midi notes)"""
    chords_note = np.array([-1, -1])
    bass_note = -1
    melody_note = 48
    walking_note = 36
    velocities = gen_velocities(solo)
    for i, _ in enumerate(solo):
        #print(i, melody[i])
        if melody_note != 48:
            PLAYER.note_off(melody_note+SOLO_OFFSET, velocities[i], SOLO_CHAN)
        melody_note = solo[i]
        if melody_note != 48:
            PLAYER.note_on(melody_note+SOLO_OFFSET, velocities[i], SOLO_CHAN)
            #print(melody_note)
        if i%2 == 0:
            PLAYER.note_off(walking_note+WALKING_OFFSET, 60, WALKING_CHAN)
            walking_note = walking[i//2]
            PLAYER.note_on(walking_note+WALKING_OFFSET, 60, WALKING_CHAN)
        current_bass = np.argmax(roll_bass[i])
        if current_bass != bass_note or i%8 == 0:
            PLAYER.note_off(bass_note + 48, 40, 2)
            bass_note = current_bass
            #print("bar #{}, BASS {}".format(i, bass_note))
            PLAYER.note_on(current_bass + 48, 40, 2)
        current_chords = np.where(roll_chords[i])[0]
        if current_chords.size == 0:
            current_chords = np.array([-1, -1])
        if (current_chords != chords_note).any() or i%8 == 0:
            PLAYER.note_off(chords_note[0]+60, 40, 2)
            PLAYER.note_off(chords_note[1]+60, 40, 2)
            chords_note = current_chords
            #print("bar #{}, CHORDS {}, {}".format(i, chords_note[0], chords_note[1]))
            PLAYER.note_on(chords_note[0]+60, 40, 2)
            PLAYER.note_on(chords_note[1]+60, 40, 2)
        if not swing:
            to_sleep = 30./tempo
        elif i%2 == 0:
            to_sleep = 34./tempo
        else:
            to_sleep = 26./tempo
        time.sleep(to_sleep)
    # off all notes, the very dirty way :(
    for note in range(128):
        PLAYER.note_off(note, channel=2)
        PLAYER.note_off(note, channel=WALKING_CHAN)
        PLAYER.note_off(note, channel=SOLO_CHAN)

if __name__ == "__main__":
    from gen_walking_multi import gen_walking
    from gen_melody_multi import gen_melody
    from keras.models import load_model
    from generate_melody import TEST_BASS, TEST_CHORDS
    walking_model = load_model("shallow_walking_good.h5")
    solo_model = load_model("models/multidirectional.h5")
    print("generating solo...")
    solo = gen_melody(solo_model, TEST_BASS, TEST_CHORDS)
    print("generating walking bass...")
    walking = gen_walking(walking_model, TEST_BASS[::2], TEST_CHORDS[::2])
    play_all(TEST_BASS, TEST_CHORDS, 170, solo, walking)
