import midi
import matplotlib.pyplot as plt
import argparse
import glob
import sys


for file in glob.glob(sys.argv[1]):
	song = midi.read_midifile(file.midi_file)
	song.make_ticks_abs()
	tracks = []
	for track in song:
	    notes = [note for note in track if note.name == 'Note On']
	    pitch = [note.pitch for note in notes]
	    tick = [note.tick for note in notes]
	    tracks += [tick, pitch]
	plt.plot(*tracks)

plt.show()