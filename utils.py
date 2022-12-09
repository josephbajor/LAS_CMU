import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hparams import Hparams
import time
import os
import subprocess
import wandb
import torch
import pandas as pd
import Levenshtein


def initiate_run(hparams: Hparams, model:torch.nn.Module):
    """
    Initialize connection to wandb and begin the run using provided hparams
    """
    with open(hparams.keyring_dir + "wandb.key") as key:
        wandb.login(key=key.read().strip())
        key.close()

    if hparams.use_wandb:
        mode = "online"
    else:
        mode = "disabled"

    run = wandb.init(
        name=f"{hparams.architecture}_{int(time.time())}",
        project=hparams.project,
        config=hparams.wandb_export(),
        mode=mode,
    )

    wandb.watch(model, log="all")

    wandb.config.update(  # Add model parameter count
        {"parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    )

    return run


def load_model(hparams, model, optimizer, scheduler=None):

    model_pth = os.path.join(
        hparams.model_dir, f"{hparams.architecture}/checkpoint.pth"
    )

    params = torch.load(model_pth)
    model.load_state_dict(params["model_state_dict"])
    optimizer.load_state_dict(params["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(params["scheduler_state_dict"])

        return model, optimizer, scheduler

    return model, optimizer


def prepare_instance() -> None:
    return NotImplementedError


def indices_to_chars(indices, vocab, EOS_TOKEN=29, SOS_TOKEN=0):
    tokens = []
    for i in indices: # This loops through all the indices
        if vocab[int(i)] == SOS_TOKEN: # If SOS is encountered, dont add it to the final list
            continue
        elif vocab[int(i)] == EOS_TOKEN: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[i])
    return tokens

def calc_edit_distance(data_type, predictions, y, ly, vocab, print_example= False):

    dist                = 0
    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size):

        if data_type == "toy":
            y_sliced    = indices_to_chars(y[batch_idx,0:ly[batch_idx]], vocab)
            pred_sliced = indices_to_chars(predictions[batch_idx], vocab)
            dist      += Levenshtein.distance(y_sliced, pred_sliced)

            if print_example: 
                print("Ground Truth : ", y_sliced)
                print("Prediction   : ", pred_sliced)

        if data_type == "main":
        # Strings - When you are using characters from the AudioDataset
            y_string    = ''.join(y_sliced)
            pred_string = ''.join(pred_sliced)
            dist        += Levenshtein.distance(pred_string, y_string)

            if print_example: 
                print("Ground Truth : ", y_string)
                print("Prediction   : ", pred_string)
        
    dist/=batch_size
    return dist


def calculate_levenshtein(hparams, h, y, lh, ly, decoder, labels):

    h = h.permute((1, 0, 2))

    labels = np.array(labels)

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(h, seq_lens=lh)

    distance = 0  # Initialize the distance to be 0 initially

    for i in range(beam_results.shape[0]):
        result = beam_results[i][0][: out_lens[i][0]]
        y_sample = y[i][: ly[i]]

        translated_result = "".join(labels[result])
        translated_y = "".join(labels[y_sample.detach().cpu()])

        distance += Levenshtein.distance(translated_result, translated_y)

    distance /= hparams.batch_size

    return distance


def submit_to_kaggle(hparams: Hparams, preds: list) -> None:
    """
    Submits the generated predictions to the kaggle competition
    """
    submission = pd.DataFrame({"label": preds})

    submission = submission.reset_index()

    submission.to_csv("submit.csv", index=False)

    subprocess.run(
        f'kaggle competitions submit -c 11-785-f22-hw3p2-slack -f submit.csv -m "{hparams.architecture}"',
        shell=True,
    )


def plot_attention(attention):
    # Function for plotting attention
    # You need to get a diagonal plot
    plt.clf()
    sns.heatmap(attention, cmap="GnBu")
    plt.show()


def setup_model_paths(hparams):
    # Create paths for model saving/loading
    if hparams.force_load_path == None:
        model_pth = os.path.join(
            hparams.model_dir, f"{hparams.architecture}/checkpoint.pth"
        )
    else:
        model_pth = os.path.join(hparams.force_load_path, "checkpoint.pth")

    if hparams.force_save_path == None:
        model_save_pth = model_pth
    else:
        model_save_pth = os.path.join(hparams.force_save_path, "checkpoint.pth")

    # create model save directory if it does not exist
    os.makedirs(
        os.path.join(hparams.model_dir, f"{hparams.architecture}/"), exist_ok=True
    )

    if hparams.force_save_path is not None:
        os.makedirs(hparams.force_save_path, exist_ok=True)

    return model_pth, model_save_pth
