import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hparams import Hparams
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchsummaryX import summary
import random


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

    def __init__(
        self, input_size, hidden_size, locked_dropout: bool = False, p: float = 0.3
    ):
        super(pBLSTM, self).__init__()

        self.locked_dropout = locked_dropout
        self.p = p

        self.ld = lstm_locked_dropout(p=p)

        self.blstm = nn.LSTM(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )  # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def trunc_reshape(self, x, x_lens):
        # TODO: If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        # TODO: Reduce lengths by the same downsampling factor

        x = x[:, : (x.shape[1] // 2) * 2]

        x = x.reshape((x.shape[0], x.shape[1] // 2, x.shape[2] * 2))
        # x_lens = x_lens // 2
        x_lens = torch.clamp(x_lens, max=x.shape[1])

        return x, x_lens

    def forward(self, x_packed):  # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence
        x, xl = pad_packed_sequence(x_packed, batch_first=True)

        # Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        # TODO: Pack Padded Sequence. What output(s) would you get?
        # TODO: Pass the sequence through bLSTM

        x, xl = self.trunc_reshape(x, xl)

        if self.locked_dropout:
            x = self.ld(x)

        x = pack_padded_sequence(x, xl, batch_first=True, enforce_sorted=False)

        x, (h_n, c_n) = self.blstm(x)

        # What do you return?

        return x


# class Listener(torch.nn.Module):
#     """
#     The Encoder takes utterances as inputs and returns latent feature representations
#     """

#     def __init__(self, hparams: Hparams, input_size=15):
#         super(Listener, self).__init__()
#         self.input_size = input_size

#         # The first LSTM at the very bottom
#         self.base_lstm = torch.nn.LSTM(
#             input_size=self.input_size, hidden_size=hparams.enc_hidden_size
#         )

#         self.pBLSTMs = torch.nn.Sequential(  # How many pBLSTMs are required?
#             # TODO: Fill this up with pBLSTMs - What should the input_size be?
#             # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
#             # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
#             pBLSTM(input_size=hparams.enc_hidden_size, hidden_size=hparams.enc_hidden_size * 2),
#             lstm_locked_dropout(p=hparams.p_lockdrop),
#             pBLSTM(
#                 input_size=hparams.hidden_size * 2, hidden_size=hparams.hidden_size * 4
#             ),
#             lstm_locked_dropout(p=hparams.p_lockdrop),
#             pBLSTM(
#                 input_size=hparams.hidden_size * 4, hidden_size=hparams.hidden_size * 8
#             ),
#             lstm_locked_dropout(p=hparams.p_lockdrop),
#         )

#     def forward(self, x, xl):
#         # Where are x and x_lens coming from? The dataloader

#         # TODO: Pack Padded Sequence
#         # TODO: Pass it through the first LSTM layer (no truncation)
#         # TODO: Pad Packed Sequence
#         # TODO: Pass Sequence through the pyramidal Bi-LSTM layer

#         # Remember the number of output(s) each function returns

#         x = x.permute(0,2,1)

#         x = pack_padded_sequence(x, xl, batch_first=True, enforce_sorted=False)

#         x, (h_n, c_n) = self.base_lstm(x)

#         x = self.pBLSTMs(x)

#         x, xl = pad_packed_sequence(x, batch_first=True, enforce_sorted=False)

#         return x, xl


class ModularListener(nn.Module):
    def __init__(self, hparams: Hparams, input_size: int = 15) -> None:
        super().__init__()
        self.input_size = input_size
        self.hparams = hparams

        self.embedding = nn.Conv1d(
            in_channels=15,
            out_channels=hparams.enc_init_emb_dims,
            kernel_size=5,
            padding=2,
        )

        if hparams.enc_use_conv1d_emb:
            self.base_lstm = torch.nn.LSTM(
                input_size=hparams.enc_init_emb_dims,
                hidden_size=hparams.enc_hidden_size,
                bidirectional=True,
                batch_first=True,
            )
        else:
            self.base_lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hparams.enc_hidden_size,
                bidirectional=True,
                batch_first=True,
            )

        self.pBLSTMs = torch.nn.Sequential()

        for i in range(hparams.enc_pyramidal_layers):
            self.pBLSTMs.add_module(
                f"p_BLSTM-{i+1}",
                pBLSTM(
                    input_size=hparams.enc_hidden_size * (2**i) * 2,
                    hidden_size=hparams.enc_hidden_size * (2 ** (i + 1)),
                    locked_dropout=self.hparams.enc_locked_dropout,
                    p=self.hparams.enc_p_lockdrop,
                ),
            )

    def forward(self, x, xl):

        if self.hparams.enc_use_conv1d_emb:
            x = x.permute(0, 2, 1)
            x = self.embedding(x)
            x = x.permute(0, 2, 1)

        x = pack_padded_sequence(x, xl, batch_first=True, enforce_sorted=False)

        x, (h_n, c_n) = self.base_lstm(x)

        x = self.pBLSTMs(x)

        x, xl = pad_packed_sequence(x, batch_first=True)

        return x, xl


class Attention(torch.nn.Module):
    """
    Attention is calculated using the key, value (from encoder hidden states) and query from decoder.
    Here are different ways to compute attention and context:

    After obtaining the raw weights, compute and return attention weights and context as follows.:

    masked_raw_weights  = mask(raw_weights) # mask out padded elements with big negative number (e.g. -1e9 or -inf in FP16)
    attention           = softmax(masked_raw_weights)
    context             = bmm(attention, value)

    At the end, you can pass context through a linear layer too.

    """

    def __init__(self, hparams: Hparams, DEVICE):
        super(Attention, self).__init__()

        self.hparams = hparams

        self.DEVICE = DEVICE

        self.key_projection = nn.Linear(
            self.hparams.enc_output_size, self.hparams.att_projection_size
        )
        self.value_projection = nn.Linear(
            self.hparams.enc_output_size, self.hparams.att_projection_size
        )
        self.query_projection = nn.Linear(
            self.hparams.dec_output_size, self.hparams.att_projection_size
        )

        self.softmax = nn.Softmax(1)

    # As you know, in the attention mechanism, the key, value and mask are calculated only once.
    # This function is used to calculate them and set them to self
    def set_key_value_mask(self, encoder_outputs, encoder_lens):

        _, encoder_max_seq_len, _ = encoder_outputs.shape

        self.key = self.key_projection(encoder_outputs)
        self.value = self.value_projection(encoder_outputs)

        # encoder_max_seq_len is of shape (batch_size, ) which consists of the lengths encoder output sequences in that batch
        # The raw_weights are of shape (batch_size, timesteps)

        # TODO: To remove the influence of padding in the raw_weights, we want to create a boolean mask of shape (batch_size, timesteps)
        # The mask is False for all indicies before padding begins, True for all indices after.
        self.padding_mask = torch.arange(encoder_max_seq_len).unsqueeze(
            0
        ) >= encoder_lens.unsqueeze(1)
        self.padding_mask = self.padding_mask.to(self.DEVICE)

    def forward(self, decoder_output_embedding):
        # key   : (batch_size, timesteps, projection_size)
        # value : (batch_size, timesteps, projection_size)
        # query : (batch_size, projection_size)

        self.query = self.query_projection(
            decoder_output_embedding
        )  # TODO: Project the query using query_projection

        # Hint: Take a look at torch.bmm for the products below

        raw_weights = torch.bmm(self.key, self.query.unsqueeze(2)).squeeze(
            2
        )  # TODO: Calculate raw_weights which is the product of query and key, and is of shape (batch_size, timesteps)
        raw_weights.masked_fill_(
            self.padding_mask, float("-inf")
        )  # TODO: Mask the raw_weights with self.padding_mask.
        # Take a look at pytorch's masked_fill_ function (You want the fill value to be a big negative number for the softmax to make it close to 0)

        attention_weights = self.softmax(
            raw_weights / np.sqrt(self.hparams.att_projection_size)
        )  # TODO: Calculate the attention weights, which is the softmax of raw_weights
        context = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(
            1
        )  # TODO: Calculate the context - it is a product between attention_weights and value

        # Hint: You might need to use squeeze/unsqueeze to make sure that your operations work with bmm

        return context, attention_weights  # Return the context, attention_weights


class Speller(torch.nn.Module):
    def __init__(
        self,
        hparams: Hparams,
        vocab_size,
        SOS_TOKEN: int,
        EOS_TOKEN: int,
        DEVICE: str,
        attention_module: Attention = None,
    ):
        super().__init__()

        self.hparams = hparams
        self.SOS_TOKEN = SOS_TOKEN
        self.EOS_TOKEN = EOS_TOKEN
        self.DEVICE = DEVICE

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size, self.hparams.dec_emb_size, padding_idx=EOS_TOKEN
        )  # TODO: Initialize the Embedding Layer (Use the nn.Embedding Layer from torch), make sure you set the correct padding_idx

        self.lstm_cells = torch.nn.Sequential(
            # Create Two LSTM Cells as per LAS Architecture
            # What should the input_size of the first LSTM Cell?
            # Hint: It takes in a combination of the character embedding and context from attention
            nn.LSTMCell(
                self.hparams.dec_emb_size + self.hparams.att_projection_size,
                self.hparams.dec_hidden_size,
            ),
            nn.LSTMCell(self.hparams.dec_hidden_size, self.hparams.dec_output_size),
        )

        # We are using LSTMCells because process individual time steps inputs and not the whole sequence.
        # Think why we need this in terms of the query

        self.char_prob = nn.Linear(
            2 * self.hparams.att_projection_size, vocab_size
        )  # TODO: Initialize the classification layer to generate your probability distribution over all characters

        self.char_prob.weight = self.embedding.weight  # Weight tying

        self.attention = attention_module

    def forward(self, encoder_outputs, encoder_lens, y=None, tf_rate=1):

        """
        Args:
            embedding: Attention embeddings
            hidden_list: List of Hidden States for the LSTM Cells
        """

        batch_size, encoder_max_seq_len, _ = encoder_outputs.shape

        encoder_outputs = encoder_outputs.to(self.DEVICE)

        if self.training:
            timesteps = y.shape[
                1
            ]  # The number of timesteps is the sequence of length of your transcript during training
            label_embed = self.embedding(
                y
            )  # Embeddings of the transcript, when we want to use teacher forcing
        else:
            timesteps = 600  # 600 is a design choice that we recommend, however you are free to experiment.

        # INITS
        predictions = []

        # Initialize the first character input to your decoder, SOS
        char = torch.full(
            (batch_size,), fill_value=self.SOS_TOKEN, dtype=torch.long
        ).to(self.DEVICE)

        # Initialize a list to keep track of LSTM Cell Hidden and Cell Memory States, to None
        hidden_states = [None] * len(self.lstm_cells)

        attention_plot = []
        context = torch.zeros(
            (batch_size, self.hparams.att_projection_size)
        )  # TODO: Initialize context (You have a few choices, refer to the writeup )
        attention_weights = torch.zeros(
            batch_size, encoder_max_seq_len
        )  # Attention Weights are zero if not using Attend Module

        # Set Attention Key, Value, Padding Mask just once
        if self.attention != None:
            self.attention.set_key_value_mask(encoder_outputs, encoder_lens)

        for t in range(timesteps):

            char_embed = self.embedding(
                char
            )  # TODO: Generate the embedding for the character at timestep t

            if self.training and t > 0:
                # TODO: We want to decide which embedding to use as input for the decoder during training
                # We can use the embedding of the transcript character or the embedding of decoded/predicted character, from the previous timestep
                # Using the embedding of the transcript character is teacher forcing, it is very important for faster convergence
                # Use a comparison between a random probability and your teacher forcing rate, to decide which embedding to use

                if random.random() <= tf_rate:
                    char_embed = label_embed[:, t - 1, :]

            decoder_input_embedding = torch.cat(
                (char_embed.to(self.DEVICE), context.to(self.DEVICE)), dim=1
            )  # TODO: What do we want to concatenate as input to the decoder? (Use torch.cat)

            # Loop over your lstm cells
            # Each lstm cell takes in an embedding
            for i in range(len(self.lstm_cells)):
                # An LSTM Cell returns (h,c) -> h = hidden state, c = cell memory state
                # Using 2 LSTM Cells is akin to a 2 layer LSTM looped through t timesteps
                # The second LSTM Cell takes in the output hidden state of the first LSTM Cell (from the current timestep) as Input, along with the hidden and cell states of the cell from the previous timestep
                hidden_states[i] = self.lstm_cells[i](
                    decoder_input_embedding, hidden_states[i]
                )
                decoder_input_embedding = hidden_states[i][0]

            # The output embedding from the decoder is the hidden state of the last LSTM Cell
            decoder_output_embedding = hidden_states[-1][0]

            # We compute attention from the output of the last LSTM Cell
            if self.attention != None:
                context, attention_weights = self.attention(
                    decoder_output_embedding
                )  # The returned query is the projected query

            attention_plot.append(attention_weights[0].detach().cpu())

            output_embedding = torch.cat(
                (self.attention.query, context), dim=1
            )  # TODO: Concatenate the projected query with context for the output embedding
            # Hint: How can you get the projected query from attention
            # If you are not using attention, what will you use instead of query?

            char_prob = self.char_prob(output_embedding)

            # Append the character probability distribution to the list of predictions
            predictions.append(char_prob)

            char = char_prob.argmax(
                axis=1
            )  # TODO: Get the predicted character for the next timestep from the probability distribution
            # (Hint: Use Greedy Decoding for starters)

        attention_plot = torch.stack(
            attention_plot, dim=0
        )  # TODO: Stack list of attetion_plots
        predictions = torch.stack(predictions, dim=1)  # TODO: Stack list of predictions

        return predictions, attention_plot


class LAS(torch.nn.Module):
    def __init__(
        self,
        hparams: Hparams,
        SOS_TOKEN: int,
        EOS_TOKEN: int,
        DEVICE: str,
        vocab_size,
        input_size=15,
    ):

        super(LAS, self).__init__()

        self.encoder = ModularListener(hparams, input_size=input_size).to(DEVICE)
        attention_module = Attention(hparams, DEVICE=DEVICE).to(DEVICE)
        self.decoder = Speller(
            hparams,
            vocab_size=vocab_size,
            SOS_TOKEN=SOS_TOKEN,
            EOS_TOKEN=EOS_TOKEN,
            DEVICE=DEVICE,
            attention_module=attention_module,
        ).to(DEVICE)

    def forward(self, x, x_lens, y=None, tf_rate=1):

        encoder_outputs, encoder_lens = self.encoder(x, x_lens)  # from Listener
        predictions, attention_plot = self.decoder(
            encoder_outputs, encoder_lens, y, tf_rate
        )

        return predictions, attention_plot


if __name__ == "__main__":
    from dataloaders import ToyDataset

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", DEVICE)

    hparams = Hparams()

    # Load the toy dataset
    X_train = np.load("f0176_mfccs_train.npy")
    X_valid = np.load("f0176_mfccs_dev.npy")
    Y_train = np.load("f0176_hw3p2_train.npy")
    Y_valid = np.load("f0176_hw3p2_dev.npy")

    # This is how you actually need to find out the different trancripts in a dataset.
    # Can you think whats going on here? Why are we using a np.unique?
    VOCAB_MAP = dict(zip(np.unique(Y_valid), range(len(np.unique(Y_valid)))))
    VOCAB_MAP["[PAD]"] = len(VOCAB_MAP)
    VOCAB = list(VOCAB_MAP.keys())

    SOS_TOKEN = VOCAB_MAP["[SOS]"]
    EOS_TOKEN = VOCAB_MAP["[EOS]"]
    PAD_TOKEN = VOCAB_MAP["[PAD]"]

    Y_train = [np.array([VOCAB_MAP[p] for p in seq]) for seq in Y_train]
    Y_valid = [np.array([VOCAB_MAP[p] for p in seq]) for seq in Y_valid]

    model = LAS(hparams, vocab_size=len(VOCAB))

    model = model.to(DEVICE)

    # Toy dataset dataloader
    train_data = ToyDataset(
        "train", X_train, Y_train, X_valid, Y_valid, EOS_TOKEN=EOS_TOKEN
    )
    val_data = ToyDataset(
        "valid", X_train, Y_train, X_valid, Y_valid, EOS_TOKEN=EOS_TOKEN
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=2,
        batch_size=hparams.batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_data.collate_fn,
    )
    # val_loader   = torch.utils.data.DataLoader(val_data, num_workers= 2,
    #                                         batch_size=hparams.batch_size, pin_memory= True,
    #                                         shuffle=False, collate_fn=train_data.collate_fn)

    # Main dataset dataloaders

    for data in train_loader:
        x, y, lx, ly = data
        print(x.shape, y.shape, lx.shape, ly.shape)
        break

    summary(model, x=x, x_lens=lx, y=y)
