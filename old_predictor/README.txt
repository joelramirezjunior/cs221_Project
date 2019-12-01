# Classical Piano Composer

This project allows you to train a neural network to generate midi music files that make use of a single instrument <-currently working on making this multiinstrumental 

## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21
	* Keras
	* Tensorflow
	* h5py

## Training

To train the network you run **lstm.py**.

E.g.

```
python lstm.py
```

The network will use every midi file in './midi_songs' to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes. Done so using checkpoints. 

## Generating music

Once you have trained the network you can generate text using **predict.py**

E.g.

```
python predict.py
```
You must in the code, specify which weights file you want the program to use to make its MIDI output. For example in the code chunk
----------------------------------------------------
LINE 51 in predict.py
def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    *********************************************
    	HERE RIGGHT HERE YUP HERE
    *********************************************
    #Here is where you will designate which weights to use. Here the weights used are 'weights-improvement-10-2.9352-bigger.hdf5'. 
    #You may set this to any file you'd like, but of course, try out your most recently created set of weights. 

    model.load_weights('weights-improvement-10-2.9352-bigger.hdf5')

    return model
------------------------------------------------------


And yup, that's it! Have fun!
