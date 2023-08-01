from typing import List

import os
import json
import tqdm
from datetime import datetime

from datasets import load_dataset
import torch

from torch.utils.tensorboard import SummaryWriter

from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens import HookedTransformer

from dataset import ModelDataset
from mlp import GeluMLP, SoluMLP


def save_parameters(save_dir: str, params: dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save training parameters to a json file
    with open(f"{save_dir}/params.json", "w") as f:
        json.dump(params, f)


def train_models(
        layer_idx: int,
        hs_multiples: List[int],
        mlp_type: str, # 'gelu' or 'solu'
        num_steps: int,
        device,
        save_dir: str = 'mlps',
        pre_noise=0.0,
        post_noise=0.0,
        hidden_noise=0.0,
        ):

    # model = HookedTransformer.from_pretrained_no_processing("pythia-70m-v0")
    model = HookedTransformer.from_pretrained_no_processing("pythia-1b-v0")

    # text_data = load_dataset("NeelNanda/pile-10k", split="train")
    text_data = load_dataset("openwebtext", split="train[:10%]")
    text_dataset = tokenize_and_concatenate(text_data, model.tokenizer)

    model_dataset = ModelDataset(model, layer_idx=layer_idx, dataset=text_dataset, batch_size=8, device=device)

    if mlp_type == 'gelu':
        mlp_class = GeluMLP
    elif mlp_type == 'solu':
        mlp_class = SoluMLP
    else:
        raise ValueError(f"mlp_type must be 'gelu' or 'solu', got {mlp_type}")
    
    mlps = [mlp_class(input_size=model.cfg.d_model,
                    hidden_size=hs*4*model.cfg.d_model,
                    output_size=model.cfg.d_model).to(device)
            for hs in hs_multiples]
    optimizers = [torch.optim.AdamW(mlp.parameters(), lr=1e-4) for mlp in mlps]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps) for optimizer in optimizers]

    writer = SummaryWriter()

    for batch_idx, (pre, post) in tqdm.tqdm(enumerate(model_dataset), total=num_steps):
        # pre and post have shape (batch_size * seq_len, d_model)
        # add noise to input
        pre = pre + torch.randn_like(pre) * pre_noise
        post = post + torch.randn_like(post) * post_noise

        for mlp, optimizer, scheduler in zip(mlps, optimizers, schedulers):
            optimizer.zero_grad()
            y = mlp(pre, hidden_noise=hidden_noise)
            loss = torch.nn.functional.mse_loss(y, post)
            loss.backward()

            optimizer.step()
            scheduler.step()

            writer.add_scalar(f"loss_layer{layer_idx}/{mlp.hidden_size}", loss.item(), batch_idx)
        
        if batch_idx > num_steps:
            print("Done")
            break

    # save
    for mlp in mlps:
        torch.save(mlp.state_dict(), f"{save_dir}/mlp_{mlp.hidden_size}_layer_{layer_idx}.pt")

    writer.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hs_multiples = [1, 2, 4, 8]
    hs_multiples = [1, 2, 4]
    layers = [1, 2]

    # Create a time-stamped directory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"mlps/{timestamp}"

    for layer_idx in layers:
        # Create a dictionary of parameters
        params = {
            'layer_idx': layer_idx,
            'hs_multiples': hs_multiples,
            'mlp_type': 'gelu',
            'num_steps': 40000,
            'device': str(device),
            'pre_noise': 0.01,
            'post_noise': 0.0,
            'hidden_noise': 0.01,
            'save_dir': save_dir,
        }

        save_parameters(save_dir, params)
        train_models(**params)
