#%%
import random
import numpy as np
import torch
import torch.utils.data
import glob
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
import hparams
import os
import matplotlib.pyplot as plt
from pathlib import Path
hparams = hparams.create_hparams()
import warnings
warnings.filterwarnings('ignore')

#%%
stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

print('===========================')
print('filter length', hparams.filter_length)
print('hop_length', hparams.hop_length)
print('win_length', hparams.win_length)
print('n_mel_channels', hparams.n_mel_channels)
print('sampling_rate', hparams.sampling_rate)
print('mel_fmin', hparams.mel_fmin)
print('mel_fmax', hparams.mel_fmax)
print('===========================')

def get_mel(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    melspec = melspec.data.numpy()
    return melspec


def save_spec(data, filepath, output_filename):
    if not(os.path.exists(filepath)):
        Path(filepath).mkdir(parents=True)
    np.save(os.path.join(filepath, output_filename), data)

#%% IEMOCAP

SESS_PATH = '/data3/IEMOCAP/IEMOCAP_full_release/Session1/sentences/wav/'
for sess in list(range(1,6)):
    temp_sess_path = SESS_PATH.replace('1', str(sess))
    all_wav_path = sorted(glob.glob(temp_sess_path + '*/*.wav'))
    for wav_path in all_wav_path:
        spec = get_mel(wav_path)
        output_filename = wav_path.split('/')[-1].split('.')[0]
        spec_path = Path(wav_path.replace('IEMOCAP_full_release', 'IEMOCAP_full_release_spec'))
        spec_path = str(spec_path.parent)
        save_spec(spec, spec_path, output_filename)
    print('Finished Session ', str(sess))


#%% VCTK

VCTK_PATH = '/data3/VCTK/VCTK-Corpus/wav48/'
for wav_path in sorted(glob.glob(VCTK_PATH + '*/*.wav')):
    spec = get_mel(wav_path)
    output_filename = wav_path.split('/')[-1].split('.')[0]
    spec_path = Path(wav_path.replace('/VCTK/', '/VCTK_spec/'))
    spec_path = str(spec_path.parent)
    save_spec(spec, spec_path, output_filename)


