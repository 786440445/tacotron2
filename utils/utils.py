import numpy as np
from scipy.io.wavfile import read
import torch

from hparams import symbols2id, id2symbols, UNK_INDEX


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    # ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="\t"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def text_to_sequence(text):
    result = [symbols2id[ch] if ch in symbols2id else UNK_INDEX for ch in text]
    return result


def sequence_to_text(seq):
    result = [id2symbols[idx] for idx in seq]
    return result
