import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from hparams import Hparams
from dataloaders import ToyDataset, build_loaders
from model import LAS
from utils import plot_attention, setup_model_paths, initiate_run
from train import train, validate
import wandb


def main():
    best_lev_dist = float("inf")
    tf_rate = 1.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", DEVICE)

    # Initialize hparam carrier
    hparams = Hparams()

    ### Dataloading
    if hparams.dataset_version == "toy":
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

        train_data = ToyDataset("train", X_train, Y_train, X_valid, Y_valid, EOS_TOKEN)
        val_data = ToyDataset("valid", X_train, Y_train, X_valid, Y_valid, EOS_TOKEN)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            num_workers=hparams.num_workers,
            batch_size=hparams.batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=train_data.collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            num_workers=hparams.num_workers,
            batch_size=hparams.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=train_data.collate_fn,
        )

    if hparams.dataset_version == "main":
        VOCAB = [
            "<sos>",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "'",
            " ",
            "<eos>",
        ]

        VOCAB_MAP = {VOCAB[i]: i for i in range(0, len(VOCAB))}

        SOS_TOKEN = VOCAB_MAP["<sos>"]
        EOS_TOKEN = VOCAB_MAP["<eos>"]

        train_loader, val_loader, test_loader = build_loaders(hparams, VOCAB_MAP)

    else:
        assert AssertionError, "Dataset version must be 'toy' or 'main'!"

    model_pth, model_save_pth = setup_model_paths(hparams)

    model = LAS(
        hparams,
        SOS_TOKEN=SOS_TOKEN,
        EOS_TOKEN=EOS_TOKEN,
        DEVICE=DEVICE,
        vocab_size=len(VOCAB),
        input_size=15,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams.lr, amsgrad=True, weight_decay=5e-6
    )
    criterion = torch.nn.CrossEntropyLoss(
        reduction="none"
    )  # Why are we using reduction = 'none' ?
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    epoch_offset = 0
    if hparams.warm_start:
        params = torch.load(model_pth)
        model.load_state_dict(params["model_state_dict"])
        optimizer.load_state_dict(params["optimizer_state_dict"])
        scheduler.load_state_dict(params["scheduler_state_dict"])
        epoch_offset = params["epoch"]

    run = initiate_run(hparams, model)

    for epoch in range(epoch_offset, hparams.epochs):

        print("\nEpoch: {}/{}".format(epoch + 1, hparams.epochs))

        # Call train and validate
        running_loss, running_perplexity, attention_plot = train(
            model,
            dataloader=train_loader,
            criterion=criterion,
            scaler=scaler,
            optimizer=optimizer,
            teacher_forcing_rate=1,
            DEVICE=DEVICE,
        )
        print(
            "Epoch {}/{}: Train Loss {}, Train Perplex {}".format(
                epoch + 1, hparams.epochs, running_loss, running_perplexity
            )
        )

        valid_dist = validate(
            model,
            val_loader,
            data_type=hparams.dataset_version,
            DEVICE=DEVICE,
            VOCAB=VOCAB,
        )
        # Print your metrics
        print("Validation Levenshtein Distance: {:.07f}".format(valid_dist))

        # Plot Attention
        if hparams.use_wandb == False:
            plot_attention(attention_plot)
        else:
            wandb.log(
                {
                    "attention_map": wandb.plots.HeatMap(
                        matrix_values=attention_plot,
                        show_text=False,
                        x_labels=[i for i in range(attention_plot.shape[1])],
                        y_labels=[i for i in range(attention_plot.shape[0])],
                    )
                }
            )

        # Log metrics to Wandb
        wandb.log(
            {
                "train_loss": running_loss,
                "train_perplex": running_perplexity,
                "validation_dist": valid_dist,
                "tf_rate": tf_rate,
                "learning_Rate": optimizer.param_groups[0]["lr"],
            }
        )
        # Optional: Scheduler Step / Teacher Force Schedule Step

        if valid_dist <= best_lev_dist:
            best_lev_dist = valid_dist
            print("Saving model")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_dist": valid_dist,
                "epoch": epoch,
            },
            model_save_pth,
        )


if __name__ == "__main__":
    main()
