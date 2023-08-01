from typing import List

from datasets import load_dataset
import torch

from torch.utils.tensorboard import SummaryWriter

from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens import HookedTransformer

import tqdm

from dataset import ModelDataset
from mlp import GeluMLP, SoluMLP


def train_models(
        layer_idx: int,
        hs_multiples: List[int],
        mlp_type: str, # 'gelu' or 'solu'
        num_steps: int,
        device,
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
        torch.save(mlp.state_dict(), f"mlps/mlp_{mlp.hidden_size}_layer_{layer_idx}.pt")

    writer.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = 40000
    hs_multiples = [1, 2, 4, 8]

    for layer_idx in range(1, 4):
        train_models(
            layer_idx,
            hs_multiples=hs_multiples,
            mlp_type='gelu',
            num_steps=num_steps,
            device=device,
            pre_noise=0.0,
            post_noise=0.0,
            hidden_noise=0.0,
        )

