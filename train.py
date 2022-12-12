from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import calc_edit_distance, calculate_levenshtein


def train(
    model, dataloader, criterion, scaler, optimizer, teacher_forcing_rate, DEVICE
):

    model.train()
    batch_bar = tqdm(
        total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train"
    )

    running_loss = 0.0
    running_perplexity = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):

        optimizer.zero_grad()

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

        with torch.cuda.amp.autocast():

            predictions, attention_plot = model(
                x, lx, y=y, tf_rate=teacher_forcing_rate
            )

            # Predictions are of Shape (batch_size, timesteps, vocab_size).
            # Transcripts are of shape (batch_size, timesteps) Which means that you have batch_size amount of batches with timestep number of tokens.
            # So in total, you have batch_size*timesteps amount of characters.
            # Similarly, in predictions, you have batch_size*timesteps amount of probability distributions.
            # How do you need to modify transcipts and predictions so that you can calculate the CrossEntropyLoss? Hint: Use Reshape/View and read the docs
            loss = criterion(
                predictions.reshape(-1, predictions.shape[2]), y.reshape(-1)
            )  # TODO: Cross Entropy Loss

            mask = (
                (torch.arange(y.shape[1]).unsqueeze(0) < ly.unsqueeze(1))
                .float()
                .to(DEVICE)
            )  # TODO: Create a boolean mask using the lengths of your transcript that remove the influence of padding indices (in transcripts) in the loss
            masked_loss = torch.sum((loss * mask.reshape(-1))) / sum(
                mask.reshape(-1)
            )  # Product between the mask and the loss, divided by the mask's sum. Hint: You may want to reshape the mask too
            perplexity = torch.exp(
                masked_loss
            )  # Perplexity is defined the exponential of the loss

            running_loss += masked_loss.item()
            running_perplexity += perplexity.item()

        # Backward on the masked loss
        scaler.scale(masked_loss).backward()

        # Optional: Use torch.nn.utils.clip_grad_norm to clip gradients to prevent them from exploding, if necessary
        # If using with mixed precision, unscale the Optimizer First before doing gradient clipping

        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss / (i + 1)),
            perplexity="{:.04f}".format(running_perplexity / (i + 1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
            tf_rate="{:.02f}".format(teacher_forcing_rate),
        )
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)
    batch_bar.close()

    return running_loss, running_perplexity, attention_plot


def validate(model, dataloader, data_type, DEVICE, VOCAB, EOS_TOKEN, SOS_TOKEN):

    model.eval()

    batch_bar = tqdm(
        total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val"
    )

    running_lev_dist = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

        with torch.inference_mode():
            predictions, attentions = model(x, lx, y=None)

        # Greedy Decoding
        greedy_predictions = torch.argmax(
            predictions, axis=-1
        )  # TODO: How do you get the most likely character from each distribution in the batch?

        if i + 1 == len(dataloader):
            print_example = True
        else:
            print_example = False

        # Calculate Levenshtein Distance
        running_lev_dist += calc_edit_distance(
            data_type,
            greedy_predictions,
            y,
            ly,
            VOCAB,
            EOS_TOKEN,
            SOS_TOKEN,
            print_example=print_example,
        )  # You can use print_example = True for one specific index i in your batches if you want

        batch_bar.set_postfix(dist="{:.04f}".format(running_lev_dist / (i + 1)))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    batch_bar.close()
    running_lev_dist /= len(dataloader)

    return running_lev_dist  # , running_loss, running_perplexity,
