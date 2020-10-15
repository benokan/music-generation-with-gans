import music21
import numpy as np
import os
import shutil
import pretty_midi
from pypianoroll import Multitrack, Track
import write_midi

import tensorflow as tf

ROOT_PATH = 'C:/Users/benok/PycharmProjects/Midi/MIDI/SuperMarioMidi'
test_ratio = 1
LAST_BAR_MODE = 'remove'




def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 64) is not 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll, np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128)
    return piano_roll


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


def save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 44, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 60, bars.shape[3]))), axis=2)
    pause = np.zeros((bars.shape[0], 64, 128, bars.shape[3]))
    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
        images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
                                                                                 images_with_pause.shape[1],
                                                                                 images_with_pause.shape[2]))

    # write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[33, 0, 25, 49, 0],
    #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)
    write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[0], is_drum=[False], filename=file_path,
                                         tempo=tempo, beat_resolution=4)



"""1. divide the original set into train and test sets""" # check
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'SuperMarioMidi_midi'))]
# print(l)
# idx = np.random.choice(len(l), int(test_ratio * len(l)), replace=False)
# print(len(idx))
# for i in idx:
#     shutil.move(os.path.join(ROOT_PATH, 'SuperMarioMidi_midi', l[i]),
#                 os.path.join(ROOT_PATH, 'SuperMarioMidi_train/origin_midi', l[i]))
#

# """2. convert_clean_train.py""" # Check

"""3. choose the clean midi from original sets"""
# if not os.path.exists(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_midi')):
#     os.makedirs(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_midi'))
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner'))]
# print(l)
# print(len(l))
# for i in l:
#     shutil.copy(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/origin_midi', os.path.splitext(i)[0] + '.mid'),
#                 os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_midi', os.path.splitext(i)[0] + '.mid'))

"""4. merge and crop"""
# if not os.path.exists(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_midi_gen')):
#     os.makedirs(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_midi_gen'))
# if not os.path.exists(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_npy')):
#     os.makedirs(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_npy'))
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_midi'))]
# print(l)
# count = 0
# for i in range(len(l)):
#     try:
#         multitrack = Multitrack(beat_resolution=4, name=os.path.splitext(l[i])[0])
#         x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_midi', l[i]))
#         multitrack.parse_pretty_midi(x)
#
#         category_list = {'Piano': [], 'Drums': []}
#         program_dict = {'Piano': 0, 'Drums': 0}
#
#         for idx, track in enumerate(multitrack.tracks):
#             if track.is_drum:
#                 category_list['Drums'].append(idx)
#             else:
#                 category_list['Piano'].append(idx)
#         tracks = []
#         merged = multitrack[category_list['Piano']].get_merged_pianoroll()
#         print(merged.shape)
#
#         pr = get_bar_piano_roll(merged)
#         print(pr.shape)
#         pr_clip = pr[:, :, 60:84]
#         print(pr_clip.shape)
#         if int(pr_clip.shape[0] % 4) != 0:
#             pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
#         pr_re = pr_clip.reshape(-1, 64, 24, 1)
#
#         save_midis(pr_re, os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_midi_gen', os.path.splitext(l[i])[0] +
#                                        '.mid'))
#         np.save(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_npy', os.path.splitext(l[i])[0] + '.npy'), pr_re)
#         print('SAVED SONG {}'.format(i))
#     except Exception as e:
#         count += 1
#         print(e, l[i])
#         continue
# print(count)


"""5. concatenate into a big binary numpy array file"""
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_npy'))]
# print(l)
# songIndices = []
# train = np.load(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_npy', l[0]))
# print(train.shape, np.max(train))
# for i in range(1, len(l)):
#     print(i, l[i])
#     t = np.load(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/cleaner_npy', l[i]))
#     train = np.concatenate((train, t), axis=0)
#     print(train.shape)
#     print(len(train))
#     songIndices.append(len(train))
# np.save(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/rock_train_piano.npy'), (train > 0.0))
# np.save("songIndices.npy",songIndices)


"""6. separate numpy array file into single phrases"""
# if not os.path.exists(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/phrase_train')):
#     os.makedirs(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/phrase_train'))
# x = np.load(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/rock_train_piano.npy'))
# count = 0
#
#
# for i in range(x.shape[0]):
#     if np.max(x[i]):
#         count += 1
#         np.save(os.path.join(ROOT_PATH, 'SuperMarioMidi_train/phrase_train/rock_piano_train_{}.npy'.format(i+1)), x[i])
#
#

# full_roll = np.load('MIDI/SuperMarioMidi/SuperMarioMidi_train/rock_train_piano.npy')
indices = np.load('songIndices.npy')
# print(indices)
# print(full_roll.shape)
# songList = []
#
# songList.append(full_roll[:indices[0]])
# print(songList[0].shape)
#
# for i in range(0,len(indices)):
#     songList.append(full_roll[indices[i]:indices[i+1]])
#     if i == 868:
#         break
# print(np.shape(songList[1]))
#
# np.save('songListSeperated.npy',songList)

songListSeperated = np.load('songListSeperated.npy',allow_pickle=True)




# for i in range(0,len(indices)):
#     np.save('song{}'.format(i),)

# s4_song1 = s4[:24]
# s4_song2 = s4[24:48]
# s4_song3 = s4[48:68]
# s4_song4 = s4[68:92]
# s4_song5 = s4[92:112]
# s4_song6 = s4[112:132]
# s4_song7 = s4[132:152]
# s4_song8 = s4[152:164]
# s4_song9 = s4[164:188]
# s4_song10 = s4[188:208]
# s4_song11 = s4[208:228]

