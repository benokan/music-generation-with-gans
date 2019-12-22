import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np  
import random



train_path = 'data/songListSeperated.npy'
data = np.load(train_path)
dataset = []
for d in data:
    # print(d.shape)
    track = d.reshape((-1, 16, 24, 1))
    track = np.concatenate((np.zeros((track.shape[0], track.shape[1], 60, 1)), track, np.zeros((track.shape[0], track.shape[1], 44, 1))), axis=2)
    prev_bars = track.copy()
    prev_bars = np.delete(prev_bars, -1, 0)
    z = np.zeros((16,128,1))
    prev_bars = np.insert(prev_bars, 0, z, axis=0)
    for t, p in zip(track, prev_bars):
        dataset.append((t,p))

random.shuffle(dataset)
dataset = np.array(dataset)
print(dataset.shape)
np.save("piano_data_24", dataset)
