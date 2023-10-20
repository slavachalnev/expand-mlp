# Adapted from https://github.com/HoagyC/sparse_coding
# Mostly using it as-is

import asyncio
import copy
import importlib
import json
import multiprocessing as mp
import os
import pickle
import sys
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from datasets import load_dataset
from transformer_lens import HookedTransformer

mp.set_start_method("spawn", force=True)

from neuron_explainer.activations.activation_records import \
    calculate_max_activation
from neuron_explainer.activations.activations import (
    ActivationRecord, ActivationRecordSliceParams, NeuronId, NeuronRecord)
from neuron_explainer.explanations.calibrated_simulator import \
    UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import \
    TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import (
    aggregate_scored_sequence_simulations, simulate_and_score)
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.fast_dataclasses import loads

from mlp import ReluMLP, MLP


EXPLAINER_MODEL_NAME = "gpt-4"  # "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003"

OPENAI_MAX_FRAGMENTS = 50000
OPENAI_FRAGMENT_LEN = 64
OPENAI_EXAMPLES_PER_SPLIT = 5
N_SPLITS = 4
TOTAL_EXAMPLES = OPENAI_EXAMPLES_PER_SPLIT * N_SPLITS
REPLACEMENT_CHAR = "�"
MAX_CONCURRENT = None


def make_tensor_name(layer: int, layer_loc: str) -> str:
    """Make the tensor name for a given layer and model."""
    assert layer_loc in [
        "residual",
        "mlp",
        "attn",
        "mlpout",
        "mlpin",
    ], f"Layer location {layer_loc} not supported"

    if layer_loc == "residual":
        tensor_name = f"blocks.{layer}.hook_resid_post"
    elif layer_loc == "mlp":
        tensor_name = f"blocks.{layer}.mlp.hook_post"
    elif layer_loc == "attn":
        tensor_name = f"blocks.{layer}.hook_resid_post"
    elif layer_loc == "mlpout":
        tensor_name = f"blocks.{layer}.hook_mlp_out"
    elif layer_loc == "mlpin":
        tensor_name = f"blocks.{layer}.ln2.hook_normalized"

    return tensor_name


def make_feature_activation_dataset(
    model: HookedTransformer,
    learned_dict: MLP,
    layer: int,
    layer_loc: str,
    device: str = "cpu",
    n_fragments=OPENAI_MAX_FRAGMENTS,
    max_features: int = 0,  # number of features to store activations for, 0 for all
    random_fragment=True,  # used for debugging
):
    """
    Takes a specified point of a model, and a dataset.
    Returns a dataset which contains the activations of the model at that point,
    for each fragment in the dataset, transformed into the feature space
    """
    model.to(device)
    model.eval()
    learned_dict.to_device(device)

    if max_features:
        feat_dim = min(max_features, learned_dict.hidden_size)
    else:
        feat_dim = learned_dict.hidden_size

    sentence_dataset = load_dataset("openwebtext", split="train", streaming=True)

    tokenizer_model = model

    tensor_name = make_tensor_name(layer, layer_loc)
    # make list of sentence, tokenization pairs

    iter_dataset = iter(sentence_dataset)

    # Make dataframe with columns for each feature, and rows for each sentence fragment
    # each row should also have the full sentence, the current tokens and the previous tokens

    n_thrown = 0
    n_added = 0
    batch_size = min(20, n_fragments)

    fragment_token_ids_list = []
    fragment_token_strs_list = []

    activation_maxes_table = np.zeros((n_fragments, feat_dim), dtype=np.float16)
    activation_data_table = np.zeros((n_fragments, feat_dim * OPENAI_FRAGMENT_LEN), dtype=np.float16)
    with torch.no_grad():
        while n_added < n_fragments:
            fragments: List[torch.Tensor] = []
            fragment_strs: List[str] = []
            while len(fragments) < batch_size:
                print(
                    f"Added {n_added} fragments, thrown {n_thrown} fragments\t\t\t\t\t\t",
                    end="\r",
                )
                sentence = next(iter_dataset)
                # split the sentence into fragments
                sentence_tokens = tokenizer_model.to_tokens(sentence["text"], prepend_bos=False).to(device)
                n_tokens = sentence_tokens.shape[1]
                # get a random fragment from the sentence - only taking one fragment per sentence so examples aren't correlated]
                if random_fragment:
                    token_start = np.random.randint(0, n_tokens - OPENAI_FRAGMENT_LEN)
                else:
                    token_start = 0
                fragment_tokens = sentence_tokens[:, token_start : token_start + OPENAI_FRAGMENT_LEN]
                token_strs = tokenizer_model.to_str_tokens(fragment_tokens[0])
                if REPLACEMENT_CHAR in token_strs:
                    n_thrown += 1
                    continue

                fragment_strs.append(token_strs)
                fragments.append(fragment_tokens)

            tokens = torch.cat(fragments, dim=0)
            assert tokens.shape == (batch_size, OPENAI_FRAGMENT_LEN), tokens.shape

            _, cache = model.run_with_cache(tokens)
            mlp_activation_data = cache[tensor_name].to(device)

            for i in range(batch_size):
                fragment_tokens = tokens[i : i + 1, :]
                activation_data = mlp_activation_data[i : i + 1, :].squeeze(0)
                token_ids = fragment_tokens[0].tolist()

                feature_activation_data = learned_dict.encode(activation_data)
                feature_activation_maxes = torch.max(feature_activation_data, dim=0)[0]

                activation_maxes_table[n_added, :] = feature_activation_maxes.cpu().numpy()[:feat_dim]

                feature_activation_data = feature_activation_data.cpu().numpy()[:, :feat_dim]

                activation_data_table[n_added, :] = feature_activation_data.flatten()

                fragment_token_ids_list.append(token_ids)
                fragment_token_strs_list.append(fragment_strs[i])

                n_added += 1

                if n_added >= n_fragments:
                    break

    print(f"Added {n_added} fragments, thrown {n_thrown} fragments")
    # Now we build the dataframe from the numpy arrays and the lists
    print(f"Making dataframe from {n_added} fragments")
    df = pd.DataFrame()
    df["fragment_token_ids"] = fragment_token_ids_list
    df["fragment_token_strs"] = fragment_token_strs_list
    maxes_column_names = [f"feature_{i}_max" for i in range(feat_dim)]
    activations_column_names = [
        f"feature_{i}_activation_{j}" for j in range(OPENAI_FRAGMENT_LEN) for i in range(feat_dim)
    ]  # nested for loops are read left to right

    assert feature_activation_data.shape == (OPENAI_FRAGMENT_LEN, feat_dim)
    df = pd.concat([df, pd.DataFrame(activation_maxes_table, columns=maxes_column_names)], axis=1)
    df = pd.concat(
        [df, pd.DataFrame(activation_data_table, columns=activations_column_names)],
        axis=1,
    )
    print(f"Threw away {n_thrown} fragments, made {len(df)} fragments")
    return df


def get_df(
    feature_dict: MLP,
    model_name: str,
    layer: int,
    layer_loc: str,
    n_feats: int,
    save_loc: str,
    device: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    # Load feature dict
    feature_dict.to_device(device)

    df_loc = os.path.join(save_loc, f"activation_df.hdf")

    reload_data = True
    if os.path.exists(df_loc) and not force_refresh:
        start_time = datetime.now()
        base_df = pd.read_hdf(df_loc)
        print(f"Loaded dataset in {datetime.now() - start_time}")

        # Check that the dataset has enough features saved
        if f"feature_{n_feats - 1}_activation_0" in base_df.keys():
            reload_data = False
        else:
            print("Dataset does not have enough features, remaking")

    if reload_data:
        model = HookedTransformer.from_pretrained(model_name, device=device)

        base_df = make_feature_activation_dataset(
            model,
            learned_dict=feature_dict,
            layer=layer,
            layer_loc=layer_loc,
            device=device,
            max_features=n_feats,
        )
        # save the dataset, saving each column separately so that we can retrive just the columns we want later
        print(f"Saving dataset to {df_loc}")
        os.makedirs(save_loc, exist_ok=True)
        base_df.to_hdf(df_loc, key="df", mode="w")

    # save the autoencoder being investigated
    os.makedirs(save_loc, exist_ok=True)
    torch.save(feature_dict, os.path.join(save_loc, "autoencoder.pt"))

    return base_df
