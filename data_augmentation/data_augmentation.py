import librosa
import os
import numpy as np
import glob
from glob import iglob
import matplotlib.pyplot as plt
import librosa.display
import random

n_songs = 100
per_song_Augmntatn = 1
rootdir_glob = "../data/genres/**"
save_path = "../data/genres/"
count = 0
def manipulate(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift(data, sampling_rate, shift_max, shift_direction):
    shift = sampling_rate*shift_max
    shift_data = [0]*shift
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data=np.concatenate((augmented_data, shift_data), axis=0)
    else:
        augmented_data = np.concatenate((shift_data,augmented_data),axis=0)
    return augmented_data

def pitch_shift(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def speed_change(data, speed_factor,sampleRate):
    data = librosa.effects.time_stretch(data, speed_factor)
    data = shift(data,sampleRate,5,'both') #Shifting the song to make it 30 secs long.
    return data

for filename in glob.iglob(rootdir_glob, recursive=True):
    if os.path.isfile(filename):
        x, sampleRate = librosa.load(filename, duration=30)
        save_folder = filename.split('/')[3] #Get the genres folder
        name = "Augmented." + filename.split('/')[4]
        save_fileName = save_path  + save_folder + "/" + name
        augment_option = random.randint(1, 3)
        if(augment_option==1):
            noise_factor = np.random.normal(0,2)/100
            augment_data = manipulate(x,noise_factor)
            #librosa.output.write_wav("/Users/jindeshubham/PycharmProjects/music/genres/noise_1.wav", augment_data,
            #                         sampleRate)
        elif(augment_option==2):
            pitch_factor = np.random.normal(0,2)/100
            augment_data = pitch_shift(x,sampleRate,pitch_factor)
            #librosa.output.write_wav("/Users/jindeshubham/PycharmProjects/music/genres/pitch_1.wav", augment_data,
            #                         sampleRate)
        elif(augment_option==3):
            augment_data = speed_change(x,1.2,sampleRate)
            #librosa.output.write_wav("/Users/jindeshubham/PycharmProjects/music/genres/speed_1.wav", augment_data,
            #                          sampleRate)

        librosa.output.write_wav(save_fileName, augment_data, sampleRate)
        count = count + 1
        if(count % 10 == 0):
            print("Completed ",count," songs Augmentation")
print("Done...")
