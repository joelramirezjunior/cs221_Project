#!/usr/bin/env python
# coding: utf-8

# In[1]:

from metrics import *
import mido
import os
import glob
import pickle
from mido import MidiFile, MidiTrack, Message
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras import optimizers
from random import randint 
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


def grabName(key):
    x = key.split('/')
    y = x[1]
    z = y.split('.')
    return z[0]


# In[3]:


def geMidiFileAndTest():
    allMidiFile = 'clean_midi_songs/*.mid'
    return allMidiFile


# In[4]:


def getNotes(allMidiFile):
        training_notes = []
        prediction_notes = dict()
        
        #getting all the notes we are training model on TO MAKE prediction
        for file in glob.glob(allMidiFile):
            individualSong = []
            midi= MidiFile(file)
            print('Loading file: ', file)
            for msg in midi:
                if not msg.is_meta and msg.type == 'note_on':
                    data = msg.bytes()
                    individualSong.append(data[1])
            # we store these to make predictions based on different songs
            prediction_notes[file] = individualSong
            # these we use to train. Stored as individual list so that the songs are 
            # not overlaped in training
            training_notes.append(individualSong)
    
        return training_notes, prediction_notes


# In[5]:


# subsample data for training and prediction
def prepareSequences(n_prev, training_notes):

    sequences = []
    y = []

    #here we are iterating through different songs
    # this assures that none of the sequences have different songs
    for training_note in training_notes:
        for i in range(len(training_note)-n_prev):
            sequences.append(training_note[i:i+n_prev])
            y.append(training_note[i+n_prev])
    
    # save a seed to do prediction later
    # this will grab all elements up to the 301st to last
    print("seq", len(sequences))
    sequence_test = sequences[-300:][600:]
    sequences = np.asarray(sequences[:-300][600:])
    y = y[:-300][600:]
    # sequence_test = sequences[-300:]
    # sequences = np.asarray(sequences[:-300])
    # y = y[:-300]

    return sequences, sequence_test, y


# In[6]:


def encondeClassValues(y):
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)

    # convert integers to dummy variables (i.e. one hot encoded ;-) 
    dummy_y = np_utils.to_categorical(encoded_Y)
    return encoder, encoded_Y, dummy_y


# In[7]:


# define baseline model
def baseline_model(dummy_y, n_prev):
    print("Dummy_y, n_prev", len(dummy_y[0]), n_prev)
    # create model
    model = Sequential()

    # put in this Dense 124, but it could be easily removed.
    model.add(Dense(124, input_dim=n_prev, activation='relu'))
    model.add(Dense(124, input_dim=n_prev, activation='relu'))
    model.add(Dense(64, input_dim=n_prev, activation='relu'))
    model.add(Dense(64, input_dim=n_prev, activation='relu'))
    model.add(Dense(len(dummy_y[0]), activation='softmax'))

    # Compile model
    # Here we use a really small learning rate becuase a larger one will not converge at all
    sgd = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[8]:


def runModel(dummy_y, sequences, nprev, numIteration ):
    model = baseline_model(dummy_y, nprev)
    history = model.fit(sequences, dummy_y, 32, numIteration, verbose=1)
    # comment out if you would like to plot loss
    # plt.plot(history.history['loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train'], loc='upper left')
    return model


# In[9]:

def makePrediction(y, prediction_notes, n_prev, model):

    # these are the names for the respective file names which we created in getNotes (line 31)
    all_keys = prediction_notes.keys()

    # These are dicts inorder to predict based on different songs. This helps so that you don't 
    # have to change every single 'predicted' song in code every time. Will iterate through all.
    predictions = dict()
    class_labels = dict()
    class_labels_min = dict()
    final_labels = dict()

    # here we iterate through all songs
    for key in all_keys:
        sequenced_test_notes = []
        #differnce prediction based on different song (i.e. key)
        sequence = prediction_notes[key]
        for i in range(len(sequence)-n_prev):
            sequenced_test_notes.append(sequence[i:i+n_prev])


        # results based on song
        prediction = model.predict(np.array(sequenced_test_notes))
        class_label = np.argmax(prediction, axis=1)
        class_label_min = np.argmin(prediction, axis=1)
        labels = list(set(y))
        final_label = [labels[i] for i in class_label]
        final_labels[key] = final_label

        # saving to dictionaries
        predictions[key] = prediction
        final_labels[key] = final_label
        class_labels[key] = class_label
        class_labels_min[key] = class_label_min

    return predictions, class_labels, class_labels_min, final_labels, all_keys


# In[10]:


def createPredictedSong(final_labels, key):
    
    mid = MidiFile()
    track = MidiTrack()
    # there is some way to moderate this to make notes less clutterd. Need To Fix.
    t = 0
    for note in final_labels[key]:
        # 147 means note_on
        # 67 is velosity
        singleNote = np.asarray([147, note, 67])
        bytes = singleNote.astype(int)
        msg = Message.from_bytes(bytes[0:3])
        t += 1 
        msg.time = t
        track.append(msg)
    # created this function to make this more legible
    key = grabName(key)
    mid.tracks.append(track)
    mid.save('created/allTrainedwith-%s-asPredicted.mid' %key)


# In[11]:


def createSongUsedToPredict(prediction_notes, key):
    #exact same as above function, only used for predictors
    test_mid = MidiFile()
    test_track = MidiTrack()
    t = 0
    overlap = 0
    for note in prediction_notes[key]:
        # 147 means note_on
        # 67 is velosity
        note = np.asarray([147, note, 67])
        bytes = note.astype(int)
        msg = Message.from_bytes(bytes[0:3])
        t += 1 
        overlap+=1
        msg.time = t
        test_track.append(msg)
    test_mid.tracks.append(test_track)
    key = grabName(key)
    test_mid.save('predicted/%s.mid' %key)


# In[12]:
def plot_confusion_matrix(prediction_notes, final_labels, key):
    matrix = confusion_matrix(prediction_notes[key][10:150],
                              final_labels[key][10:150])
    df_cm = pd.DataFrame(matrix)
    plt.title(key)
    fig, ax = plt.subplots(figsize=(160, 10))
    sn.heatmap(df_cm, annot=True, linewidths=0.25, ax=ax)


# here are all the function calls
# what do you want the sequence length to be?
nprev = 4

# how many iterations do you want? (epochs)
numIteration = 100

allMidiFile = geMidiFileAndTest()
training_notes, prediction_notes = getNotes(allMidiFile)
sequences, sequence_test, y = prepareSequences(nprev, training_notes)
encoder, encoded_Y, dummy_y = encondeClassValues(y)
model = runModel(dummy_y, sequences, nprev, numIteration)
predictions, class_labels, class_labels_min, final_labels, keys = makePrediction(y, prediction_notes, nprev, model)

for key in keys:
    createPredictedSong(final_labels, key)
    createSongUsedToPredict(prediction_notes, key)
    # comment out if you would like to plot the confusion matrix
    #plot_confusion_matrix(prediction_notes, final_labels, key)

stats()

plt.show()


