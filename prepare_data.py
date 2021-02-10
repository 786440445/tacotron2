import os
import itertools
import multiprocessing
import subprocess

import pandas as pd
from g2pM import G2pM
from tqdm import tqdm
from collections import Counter

from hparams import create_hparams


home_dir = os.getcwd()

def g2p_transfer(line):
    return ' '.join(g2pmodel(line, tone=True, char_split=False))


def do_single_file_trans(input_file, save_file, sample_rate=16000, target_extension="wav"):
    #if original_extension in ["wav", "mp3"] and target_extension in ["wav", "mp3"]:
    #    sox_cmd = "sox %s -r %s -c 1 %s" % (input_file, sample_rate, save_file)
    #    subprocess.check_call(sox_cmd, shell=True)
    #else:
    #    # use ffmpeg to transform it to pcm, then use sox transform it to wav
    if target_extension in ["pcm", "raw"]:
        ffmpeg_cmd = "ffmpeg -y -i %s -ac 1 -ar %d -f s16le %s > /dev/null 2>&1" % \
                        (input_file, sample_rate, save_file)
        subprocess.call(ffmpeg_cmd, shell=True)
    elif target_extension == "wav":
        # create tmp folder for saving temp raw file, then transformed to wav, finally remove tmp file
        tmp_file = save_file[:-3] + "raw"
        ffmpeg_cmd = "ffmpeg -y -i %s -ac 1 -ar %d -f s16le %s > /dev/null 2>&1" % \
                        (input_file, sample_rate, tmp_file)
        subprocess.call(ffmpeg_cmd, shell=True)

        sox_cmd = "sox -r %d -c 1 -e signed-integer -b 16 %s %s > /dev/null 2>&1" % \
                    (sample_rate, tmp_file, save_file)
        subprocess.call(sox_cmd, shell=True)

        os.remove(tmp_file)
    else:
        ffmpeg_cmd = "ffmpeg -y -i %s %s > /dev/null > 2>&1" % (input_file, save_file)
        subprocess.call(ffmpeg_cmd, shell=True)
    pass


def get_split_corpus(type, wavids, texts):
    if type == 'train':
        data  = ''
        train_wavids = wavids[:int(0.85 * len(wavids))]
        train_texts = texts[:int(0.85 * len(wavids))]
        print('train length : ', len(train_wavids))

        for wavid, text in zip(train_wavids, train_texts):
            wavid = wavid.split('/')[1][:-4] + '.wav'
            text = ''.join(text.split(' '))
            data += wavid + '\t' + g2p_transfer(text) + '\n'
        with open(hparams.training_files, 'w', encoding='utf-8') as f:
            f.writelines(data)

    elif type == 'test':
        data  = ''
        test_wavids = wavids[int(0.85 * len(wavids)): int(0.95 * len(wavids))]
        test_texts = texts[int(0.85 * len(wavids)): int(0.95 * len(wavids))]
        print('test length : ', len(test_wavids))

        for wavid, text in zip(test_wavids, test_texts):
            wavid = wavid.split('/')[1][:-4] + '.wav'
            text = ''.join(text.split(' '))
            data += wavid + '\t' + g2p_transfer(text) + '\n'
        with open(hparams.testing_files, 'w', encoding='utf-8') as f:
            f.writelines(data) 
    
    elif type == 'dev':
        data  = ''
        dev_wavids = wavids[int(0.95 * len(wavids)):]
        dev_texts = texts[int(0.95 * len(wavids)):]
        print('dev length : ', len(dev_wavids))

        for wavid, text in zip(dev_wavids, dev_texts):
            wavid = wavid.split('/')[1][:-4] + '.wav'
            text = ''.join(text.split(' '))
            data += wavid + '\t' + g2p_transfer(text) + '\n'
        with open(hparams.validation_files, 'w', encoding='utf-8') as f:
            f.writelines(data) 
    else:
        print('type error')
        exit()


def clean_data():
    for wavfile in tqdm(os.listdir(origin_data_dir)):
        if wavfile[-4:] == '.mp3':
            file_path = os.path.join(origin_data_dir, wavfile)
            save_file = os.path.join(data_dir, wavfile.replace('mp3', 'wav'))
            do_single_file_trans(file_path, save_file, sample_rate=sample_rate, target_extension="wav")
    
    text_pd = pd.read_csv(origin_text_dir, header=None, sep='\t')
    wavids = text_pd.iloc[:, 0].values
    texts = text_pd.iloc[:, 1].values

    l = len(wavids)
    print('data length: ', l)
    get_split_corpus('train', wavids, texts)
    get_split_corpus('test', wavids, texts)
    get_split_corpus('dev', wavids, texts)
    print('data length: ', len(wavids))


def build_pinyin_vocab():
    py_total_list = []
    for file in os.listdir(os.path.join(home_dir, 'filelists')):
        pd_data = pd.read_csv(os.path.join(home_dir, 'filelists', file), header=None, sep='\t')
        py_list = pd_data.iloc[:, 1].values
        py_list = [list(s) for item in py_list for s in item.split(' ')]
        py_total_list.extend(py_list)

    py_total_list = list(itertools.chain(*py_total_list))
    dict = Counter(py_total_list)
    vocab_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    vocab_list = [item for item, count in vocab_list if count > 5]
    vocab_list = ['PAD', 'UNK', ' '] + vocab_list
    with open(os.path.join(home_dir, 'py_vocab.txt'), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(vocab_list))


if __name__ == '__main__':
    sample_rate = 22050
    hparams = create_hparams()
    g2pmodel = G2pM()

    origin_data_dir = hparams.origin_data_dir
    origin_text_dir = hparams.origin_text_dir
    data_dir = hparams.data_dir
    text_dir = hparams.text_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    # clean_data()
    build_pinyin_vocab()