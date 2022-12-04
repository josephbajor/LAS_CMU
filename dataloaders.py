import json
import torch
from tqdm import tqdm
import glob
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from hparams import Hparams


def collate_fn(batch):

    # batch of input mfcc coefficients
    batch_mfcc = [mfcc for mfcc, transcript in batch]
    # batch of output phonemes
    batch_transcript = [transcript for mfcc, transcript in batch]

    # pad batches
    batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
    lengths_mfcc = [mfcc.shape[0] for mfcc in batch_mfcc]

    batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
    lengths_transcript = [transcript.shape[0] for transcript in batch_transcript]

    # Apply data transformations
    transforms = [
        torchaudio.transforms.TimeMasking(time_mask_param=20, p=0.3),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=8),
    ]

    for t in transforms:
        batch_mfcc_pad = t(batch_mfcc_pad)

    # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
    return (
        batch_mfcc_pad,
        batch_transcript_pad,
        torch.tensor(lengths_mfcc),
        torch.tensor(lengths_transcript),
    )


def collate_fn_test(batch):
    batch_mfcc_pad = pad_sequence(batch, batch_first=True)
    lengths_mfcc = [mfcc.shape[0] for mfcc in batch]

    return batch_mfcc_pad, torch.tensor(lengths_mfcc)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, hparams: Hparams, type="train"):
        """
        Initializes the dataset.
        """
        self.hparams = hparams

        # Load the directory and all files in them
        self.mfccs, self.transcripts = [], []

        self.train_subset = self.hparams.train_subset
        self.cps_norm = self.hparams.cps_norm

        # Load phonetic transcription from json
        # This helps convert CMUDict to ARPAbet
        with open("phonetics.json") as f:
            self.mapping = json.load(f)

        self.PHONEMES = list(self.mapping.keys())
        self.LABELS = list(self.mapping.values())
        self.phone_map = {phone: idx for idx, phone in enumerate(self.PHONEMES)}

        if type == "train":

            if self.train_subset:
                self.train_dirs: list = ["train-clean-100"]
            else:
                self.train_dirs: list = ["train-clean-100", "train-clean-360"]

        if type == "val":
            self.train_dirs: list = ["dev-clean"]

        for train_set in self.train_dirs:
            self.mfcc_dir = os.path.join(hparams.data_dir, train_set, "mfcc")
            self.transcript_dir = os.path.join(
                hparams.data_dir, train_set, "transcript", "raw"
            )

            for mfcc, transcript in tqdm(
                zip(
                    sorted(glob.glob(os.path.join(self.mfcc_dir, "*.npy"))),
                    sorted(glob.glob(os.path.join(self.transcript_dir, "*.npy"))),
                )
            ):
                # Load mfcc and apply normalization
                mfcc_sample = np.load(mfcc, allow_pickle=True)
                if self.cps_norm:
                    normalized_mfcc = mfcc_sample - np.mean(mfcc_sample, axis=0)
                    self.mfccs.append(normalized_mfcc)
                else:
                    self.mfccs.append(mfcc_sample)

                # Load transcripts and convert to interger labels
                transcript_sample = np.load(transcript, allow_pickle=True)

                # Convert the transcript labels to intergers
                for idx, i in enumerate(transcript_sample):
                    transcript_sample[idx] = self.phone_map[i]
                self.transcripts.append(transcript_sample.astype(int))

        # Set the length of the dataset
        self.length = len(self.mfccs)

    def __len__(self):

        return self.length

    def __getitem__(self, ind):

        mfcc = torch.tensor(self.mfccs[ind])
        transcript = torch.tensor(self.transcripts[ind])

        return mfcc, transcript


class AudioTestDataset(torch.utils.data.Dataset):
    def __init__(self, hparams: Hparams):
        """
        Initializes the dataset.
        """

        # Load the directory and all files in them
        self.mfccs, self.transcripts = [], []

        self.train_subset = hparams.train_subset
        self.cps_norm = hparams.cps_norm

        # Load phonetic transcription from json
        # This helps convert CMUDict to ARPAbet
        with open("phonetics.json") as f:
            self.mapping = json.load(f)

        self.PHONEMES = list(self.mapping.keys())
        self.LABELS = list(self.mapping.values())
        self.phone_map = {phone: idx for idx, phone in enumerate(self.PHONEMES)}

        self.train_dirs = ["test-clean"]

        for train_set in self.train_dirs:
            self.mfcc_dir = os.path.join(hparams.data_dir, train_set, "mfcc")

            for mfcc in tqdm(sorted(glob.glob(os.path.join(self.mfcc_dir, "*.npy")))):
                # Load mfcc and apply normalization
                mfcc_sample = np.load(mfcc, allow_pickle=True)
                if self.cps_norm:
                    normalized_mfcc = mfcc_sample - np.mean(mfcc_sample, axis=0)
                    self.mfccs.append(normalized_mfcc)
                else:
                    self.mfccs.append(mfcc_sample)

        # Set the length of the dataset
        self.length = len(self.mfccs)

    def __len__(self):

        return self.length

    def __getitem__(self, ind):

        mfcc = torch.tensor(self.mfccs[ind])
        return mfcc


def build_loaders(hparams: Hparams):

    train_data = AudioDataset(hparams, type="train")
    val_data = AudioDataset(hparams, type="val")
    test_data = AudioTestDataset(hparams)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn_test,
    )

    return train_loader, val_loader, test_loader
