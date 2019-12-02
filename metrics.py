import mido
import os
import glob
import pickle
from mido import MidiFile, MidiTrack, Message
from random import randint 
import numpy as np


# In[116]:
def geMidiFileAndTest():
    createdMidiFiles = 'created/*.mid'
    unchangedMidiFiles = 'predicted/*.mid'
    return createdMidiFiles, unchangedMidiFiles

def getKey(file, created):

    if created: 
        x = file.split('/')
        y = x[1].split('-')
        return y[1]
    else: 
        x = file.split('/')
        y = x[1].split('.')
        return y[0]


# In[117]:
def getNotes(createdMidiFiles, unchangedMidiFiles ):

        created_songs = dict()
        unchanged_songs = dict()
            
        #getting all the notes we are training model on TO MAKE prediction
        for file in glob.glob(createdMidiFiles):
            individualSong = []
            midi= MidiFile(file)
            print('Loading file: ', file)
            key = getKey(file, True)
            for msg in midi:
                if not msg.is_meta and msg.type == 'note_on':
                    data = msg.bytes()
                    individualSong.append(data[1])
            # we store these to make predictions based on different songs
            created_songs[key] = individualSong
            # these we use to train. Stored as individual list so that the songs are 
            # not overlaped in training

        for file in glob.glob(unchangedMidiFiles):
            individualSong = []
            midi= MidiFile(file)
            print('Loading file: ', file)
            key = getKey(file, False)
            for msg in midi:
                if not msg.is_meta and msg.type == 'note_on':
                    data = msg.bytes()
                    individualSong.append(data[1])
            # we store these to make predictions based on different songs
            unchanged_songs[key] = individualSong
            # these we use to train. Stored as individual list so that the songs are 
            # not overlaped in training_notes           

        return created_songs, unchanged_songs

# subsample data for training and prediction
def prepareSequences(n_prev, created_songs, unchanged_songs):

    created_sequences = dict()
    unchanged_sequences = dict()

    #here we are iterating through different songs
    # this assures that none of the sequences have different songs
    for key, value in created_songs.items():
        created = []
        for i in range(len(value)-n_prev):
            created.append(value[i:i+n_prev])
        created_sequences[key] =  created

        unchanged = []
        for i in range(len(unchanged_songs[key])-n_prev):
            unchanged.append((unchanged_songs[key])[i:i+n_prev])
        unchanged_sequences[key] = unchanged

    return created_sequences, unchanged_sequences

def compareSequences(key, created_sequences, unchanged_sequences):
    created = created_sequences[key]
    oracle = (unchanged_sequences[key])
    setSequence = set()

    for x in oracle:
        setSequence.add(tuple(x))

    correct = 0.0
    total = 0.0

    for i in range(min(len(created), len(oracle))):
        if tuple(created[i]) in setSequence:
            correct += 1
        total += 1

    stat = correct/total

    print("The acurracy for song %s is %f" %(key, stat))

def stats():
    n_prev = 4 
                        
    createdMidiFiles, unchangedMidiFiles = geMidiFileAndTest()
    created_songs, unchanged_songs = getNotes(createdMidiFiles, unchangedMidiFiles)
    created_sequences, unchanged_sequences = prepareSequences(n_prev, created_songs, unchanged_songs) 

    for key in created_sequences.keys():
        compareSequences(key, created_sequences, unchanged_sequences)
