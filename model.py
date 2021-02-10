import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from layers import Postnet, TacotronSTFT

from utils import get_mask_from_lengths
from math import sqrt
from hparams import id2symbols

class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.n_mel_channels
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / hparams.n_symbols + hparams.symbols_embedding_dim)
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)


    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        max_len = torch.max(input_lengths.data).item()

        mel_padded = mel_padded.float()
        gate_padded = gate_padded.float()
        output_lengths = output_lengths.long()
        return (text_padded, input_lengths, mel_padded, max_len, output_lengths), (mel_padded, gate_padded)

    def parse_output(self, outputs, output_lengths=None):
        # print('mel_outputs: ', outputs[0].shape)
        # print('mel_outputs: ', outputs[0])
        # print('mel_outputs_postnet: ', outputs[1])
        # print('gate_outputs: ', outputs[2])
        # print('alignments: ', outputs[3])
        # print('output_lengths', output_lengths)

        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)
        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        # print('text_inputs : ', text_inputs)
        # print('text_inputs1: ', text_inputs.numpy()[0])
        # print(''.join([id2symbols[id] for id in text_inputs.numpy()[0]]))
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        embedding_inputs = self.embedding(text_inputs).transpose(1, 2)
        # print('embedding_inputs: ', embedding_inputs.shape)
        encoder_outputs = self.encoder(embedding_inputs, text_lengths)
        # print('encoder_outputs: ', encoder_outputs)
        
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths)
        