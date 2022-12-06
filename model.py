import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hparams import Hparams
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class lstm_locked_dropout(nn.Module):
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p <= 0.0 or not self.training:
            return x

        x = x.clone()
        mask = x.new_empty(1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


class pBLSTM(torch.nn.Module):

    """
    Pyramidal BiLSTM

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input?
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    """

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(
            input_size, hidden_size, bidirectional=True
        )  # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def trunc_reshape(self, x, x_lens):
        # TODO: If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        # TODO: Reduce lengths by the same downsampling factor
        if x.shape[1] % 2 != 0:
            x = x[:, :-1]

        x = x.reshape((x.shape[0], x.shape[1] / 2, x.shape[2] * 2))
        x_lens = x_lens // 2

        return x, x_lens

    def forward(self, x_packed):  # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence
        x, xl = pad_packed_sequence(x_packed, batch_first=True, enforce_sorted=False)

        # Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        # TODO: Pack Padded Sequence. What output(s) would you get?
        # TODO: Pass the sequence through bLSTM

        x, xl = self.trunc_reshape(x, xl)
        x = pack_padded_sequence(x, xl, batch_first=True, enforce_sorted=False)

        x, (h_n, c_n) = self.blstm(x)

        # What do you return?

        return x


class Listener(torch.nn.Module):
    """
    The Encoder takes utterances as inputs and returns latent feature representations
    """

    def __init__(self, hparams: Hparams, input_size=15):
        super(Listener, self).__init__()
        self.input_size = input_size

        # The first LSTM at the very bottom
        self.base_lstm = torch.nn.LSTM(
            input_size=self.input_size, hidden_size=hparams.hidden_size
        )

        self.pBLSTMs = torch.nn.Sequential(  # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be?
            # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
            pBLSTM(input_size=hparams.hidden_size, hidden_size=hparams.hidden_size * 2),
            lstm_locked_dropout(p=hparams.p_lockdrop),
            pBLSTM(
                input_size=hparams.hidden_size * 2, hidden_size=hparams.hidden_size * 4
            ),
            lstm_locked_dropout(p=hparams.p_lockdrop),
            pBLSTM(
                input_size=hparams.hidden_size * 4, hidden_size=hparams.hidden_size * 8
            ),
            lstm_locked_dropout(p=hparams.p_lockdrop),
        )

    def forward(self, x, xl):
        # Where are x and x_lens coming from? The dataloader

        # TODO: Pack Padded Sequence
        # TODO: Pass it through the first LSTM layer (no truncation)
        # TODO: Pad Packed Sequence
        # TODO: Pass Sequence through the pyramidal Bi-LSTM layer

        # Remember the number of output(s) each function returns

        x = pack_padded_sequence(x, xl, batch_first=True, enforce_sorted=False)

        x, (h_n, c_n) = self.base_lstm(x)

        x = self.pBLSTMs(x)

        x, xl = pad_packed_sequence(x, batch_first=True, enforce_sorted=False)

        return x, xl


class ModularListener(nn.Module):
    def __init__(self, hparams: Hparams, input_size: int = 15) -> None:
        super().__init__()
        self.input_size = input_size

        self.embedding = nn.Conv1d(
            in_channels=15,
            out_channels=hparams.init_emb_dims,
            kernel_size=5,
            padding=2,
        )

        self.base_lstm = torch.nn.LSTM(
            input_size=hparams.init_emb_dims, hidden_size=hparams.hidden_size
        )

        self.pBLSTMs = torch.nn.Sequential()

        for i in range(hparams.pyramidal_layers):
            self.pBLSTMs.add_module(
                f"p_BLSTM-{i+1}",
                pBLSTM(
                    input_size=hparams.hidden_size * (2**i),
                    hidden_size=hparams.hidden_size * (2 ** (i + 1)),
                ),
            )
            if hparams.locked_dropout:
                self.pBLSTMs.add_module(
                    f"locked_dropout-{i+1}", lstm_locked_dropout(p=hparams.p_lockdrop)
                )

    def forward(self, x, xl):

        x = pack_padded_sequence(x, xl, batch_first=True, enforce_sorted=False)

        x, (h_n, c_n) = self.base_lstm(x)

        x = self.pBLSTMs(x)

        x, xl = pad_packed_sequence(x, batch_first=True, enforce_sorted=False)

        return x, xl
