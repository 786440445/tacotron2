# Tacotron2 for Mandarin

1. python3 prepare_data.py
2. python3 train.py


# Structure
Embedding(256)
Encoder:
    3 * (conv1d(size=5, out_dim=256, stride=1, dilation=1 + Bn + relu + dropout)
    Bilstm(128)

Decoder:
    Prenet: Linear(256) + relu + dropout, Linear(256) + relu + dropout
    Attention_RNN() + AttentionLayer() + decoder_RNN()
    PostNet: 
        Conv1d + BN + tanh: Mel_dim -> postnet_embedding_dim
        5 * (Conv1d + BN + tanh): postnet_embedding_dim -> postnet_embedding_dim
        Conv1d + BN: postnet_embedding_dim -> n_mel_channels


Loss:
    MSE(mel_out, mel_target) + MSE(mel_out_postnet, mel_targets)
    BCE(gate_output, gate_target)
    1. gate_target为预测结束的标志。