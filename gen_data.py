# Author: Piyush Vyas and Darshan Shinde
# Script to Generate data from recorded samples

import os
import random
import librosa
import numpy as np
from os import path


# Lists all samples
females = librosa.util.find_files('./F/', ext=['m4a'], recurse=True)
males = librosa.util.find_files('./M/', ext=['m4a'], recurse=True)
noises = librosa.util.find_files('.N/', ext=['m4a'], recurse=False)


# SNRs to use
snrs = [-5, 0, 10, 15, 25]

# Function to get multiplication factor to add noise in audio at given SNR
def getB(audio, noise, snr):
    s = np.sum(np.square(audio))
    n = np.sum(np.square(noise))
    return np.sqrt(np.power(10, -snr/10) * (s/n))

# Function to generate noisy signal and save in .wav file
def processAndStore(speech, noises, snrs):
    audio, sr_a = librosa.load(speech, sr = 8000)
    
    base_path, remaining_path = speech.split('Dataset')
    base_path = base_path + 'Dataset/PS60k'
    _, gender, country, speaker, utterance = remaining_path.split('/')
    
    gender_path = path.join(base_path, gender)
    if not path.exists(gender_path):
        os.mkdir(gender_path)
    
    country_path = path.join(gender_path, country)
    if not path.exists(country_path):
        os.mkdir(country_path)
    
    speaker_path = path.join(country_path, speaker)
    if not path.exists(path.join(speaker_path)):
        os.mkdir(speaker_path)
    
    for noise_file in noises:
        noise_name = noise_file.split('/')[-1].split('.')[0]
        noise, sr_n = librosa.load(noise_file, sr = 8000)
        N = random.randint(0, len(noise) + 1 - len(audio))
        noise = noise[N : N + len(audio)]
        
        for snr in snrs:
            b = getB(audio, noise, snr)
            noisy = audio + (b * noise)
            file_name, extension = utterance.split('.')
            file_name += '_' + noise_name + '_' + str(snr) + '.wav'
            outFile_path = path.join(speaker_path, file_name)
            librosa.output.write_wav(outFile_path, noisy, 8000)


# Function to generate dataset
def gen_sampels(list, noises, snrs):
    for speech in list:
        processAndStore(speech, noises, snrs)
        
gen_samples(females, noises, snrs)
gen_samples(males, noises, snrs)
        
        
# Gen Labels
base_path = './' # root directory where all speaker recordings are stored

# List of Samples
females = librosa.util.find_files(base_path + 'PS60k/F/', ext=['wav'], recurse=True)
males = librosa.util.find_files(base_path + 'PS60k/M/', ext=['wav'], recurse=True)

# Function to generate speaker ids
def gen_speaker_id():
    speaker_list = []
    for gender in sorted(os.listdir(base_path)):
        gender_path = path.join(base_path, gender)
        if not path.isdir(gender_path) or gender not in ['M', 'F']:
            continue
        
        for country in sorted(os.listdir(gender_path)):
            country_path = path.join(gender_path, country)
            if not path.isdir(country_path):
                continue
            for speaker in sorted(os.listdir(country_path)):
                if speaker[0] == '.':
                    continue
                speaker_path = path.join(base_path, 'PS60k', gender, country, speaker)
                speaker_list.append(speaker_path)
    id_list = [i for i in range(len(speaker_list))]
    return dict(zip(speaker_list, id_list))
    

# Function to generate labels
def gen_labels():
    speaker_id = gen_speaker_id()
    for audio in females:
        path, file_name = audio.split('/')[:-1], audio.split('/')[-1]
        path = '/'.join(path)
        id = speaker_id[path]
        label = np.zeros(60)
        label[id] = 1
        np.save(path + '/' + file_name.split('.')[0] + '.npy', label)
    for audio in males:
        path, file_name = audio.split('/')[:-1], audio.split('/')[-1]
        path = '/'.join(path)
        id = speaker_id[path]
        label = np.zeros(60)
        label[id] = 1
        np.save(path + '/' + file_name.split('.')[0] + '.npy', label)


if __name__ = '__main__':
    gen_labels()

