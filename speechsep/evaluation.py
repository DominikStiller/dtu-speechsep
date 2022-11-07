# Similar to the methods used in: (A. Défossez, N. Usunier, L. Bottou, and F. Bach, Music source separation in the waveform domain, 2019. doi)
# 1. Ablation study:
#   a) no BiLSTM
#   b) no pitch/tempo aug.
#   c) no initial weight rescaling
#   d) no 1x1 convolutions in encoder
#   e) ReLU instead of GLU
#   f) no BiLSTM, depth=7
#   g) MSE loss
#   h) no resampling
#   i) no shift trick
#   j) kernel size of 1 in decoder convolutions
#
# 2. Change the initial number of channels on the model size: [32, 48, 64], w/wo DiffQ quantized
#
# Adjust the model one by one, train the models and compare test results.
#
# Code implementation: (to be continued)
