import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import tensorflow as tf 
import numpy as np 
from network import MidiNet
import matplotlib.pyplot as plt 
from write_midi import *

tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

# model_path = 'models/paper_model1_04'
experimentNUMBER = 31
model_path = 'experiments/EXPERIMENT{}t/model/model'.format(experimentNUMBER)
n_bars = 72
n_songs = 1


def generated_songs(model_path, save_path):


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = MidiNet(sess,100, [16, 128], method='wgan')
        sess.run(tf.global_variables_initializer())
        model.restore(model_path)

        songs = []
        for s in range(n_songs):
            song = []
            prev_sample = np.zeros((1, 16, 128, 1))
        
            for b in range(n_bars):
                noise = np.random.normal(loc=0.0, scale=1.0, size=(1, 100))
                new_sample = model.generate(noise, prev_sample)
                song.append(np.squeeze(new_sample, axis=0))
                prev_sample = new_sample 
            songs.append(song)       

    # songs = np.round(songs, decimals=0)
    songs = np.array(songs)
    # print(np.max(songs))
    # print(np.min(songs))
    proper_songs = []
    songs = songs.reshape((-1, n_bars, 16, 128, 1))
    for song in songs:
        song = np.round(song, decimals=0)
        full_song = np.concatenate(song, axis=0)
        proper_songs.append(full_song)
    proper_songs = np.array(proper_songs)
    try:
        write_piano_roll_to_midi(proper_songs, save_path+'proper_song.mid')
    except:
        print("[Proper Song] Invalid note values...")

    normalized = []
    for song in songs:
        normalized_song = (song-np.min(song))/(np.max(song)-np.min(song))
        
        full_song = np.concatenate(normalized_song, axis=0)
        normalized.append(full_song)
    normalized = np.array(normalized)
    try:
        write_piano_roll_to_midi(normalized, save_path+'normalized_song.mid')
    except:
        print("[Normalized Song] Invalid note values...")
    track = songs.copy()
    track = track.reshape((-1, n_bars, 16, 128, 1))
    processed = np.zeros_like(track)
    for i_t, t in enumerate(track):
        for i_b, b in enumerate(t):
            for i_s, s in enumerate(b):
                note = np.argmax(s)
                processed[i_t][i_b][i_s][note] = 1.0
    postprocessed = []
    for song in processed:
        full_song = np.concatenate(song, axis=0)
        postprocessed.append(full_song)
    postprocessed = np.array(postprocessed)
    try:
        write_piano_roll_to_midi(postprocessed, save_path+'postprocessed.mid')
    except:
        print("[Postprocessed Song] Invalid note values...")

if __name__=='__main__':
    generated_songs(model_path, 'C:/Users/benok/OneDrive/Masaüstü/submission/')