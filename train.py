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
from mlp import GeluMLP, SoluMLP, ReluMLP


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
        skip_connection: bool = False,
        ):

    # model = HookedTransformer.from_pretrained_no_processing("pythia-70m-v0")
    # model = HookedTransformer.from_pretrained_no_processing("pythia-1b-v0")
    model = HookedTransformer.from_pretrained_no_processing("EleutherAI/pythia-70m-deduped")

    # text_data = load_dataset("NeelNanda/pile-10k", split="train")
    text_data = load_dataset("openwebtext", split="train[:10%]")
    text_dataset = tokenize_and_concatenate(text_data, model.tokenizer, max_length=48)

    model_dataset = ModelDataset(
        model,
        layer_idx=layer_idx,
        dataset=text_dataset,
        batch_size=1024, 
        device=device,
        skip_connection=skip_connection,
        )

    if mlp_type == 'gelu':
        mlp_class = GeluMLP
        raise NotImplementedError
    elif mlp_type == 'solu':
        mlp_class = SoluMLP
        raise NotImplementedError

    elif mlp_type == 'relu':
        mlp_class = ReluMLP

    else:
        raise ValueError(f"mlp_type must be 'gelu', 'solu' or 'relu' got {mlp_type}")
    
    mlps = [mlp_class(input_size=model.cfg.d_model,
                    hidden_size=hs*4*model.cfg.d_model,
                    output_size=model.cfg.d_model).to(device)
            for hs in hs_multiples]
    optimizers = [torch.optim.Adam(mlp.parameters(), lr=1e-4) for mlp in mlps]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps) for optimizer in optimizers]

    writer = SummaryWriter()

    for batch_idx, (pre, post) in tqdm.tqdm(enumerate(model_dataset), total=num_steps):
        # pre and post have shape (batch_size * seq_len, d_model)
        # add noise to input
        pre = pre + torch.randn_like(pre) * pre_noise
        post = post + torch.randn_like(post) * post_noise

        for mlp, optimizer, scheduler in zip(mlps, optimizers, schedulers):
            optimizer.zero_grad()
            y, h = mlp(pre, hidden_noise=hidden_noise)
            loss = torch.nn.functional.mse_loss(y, post)
            # l1 regularization
            loss += 5e-3 * torch.norm(h, p=1) / h.shape[0]
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
    hs_multiples = [4]
    layers = [1, 2, 3, 4, 5]

    # Create a time-stamped directory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"mlps/{timestamp}"

    for layer_idx in layers:
        # Create a dictionary of parameters
        params = {
            'layer_idx': layer_idx,
            'hs_multiples': hs_multiples,
            'mlp_type': 'relu',
            'num_steps': 100000,
            'device': str(device),
            'pre_noise': 0.0,
            'post_noise': 0.0,
            'hidden_noise': 0.0,
            'save_dir': save_dir,
            'skip_connection': False,
        }

        save_parameters(save_dir, params)
        train_models(**params)
