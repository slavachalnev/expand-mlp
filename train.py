from datasets import load_dataset
import torch

from torch.utils.tensorboard import SummaryWriter

from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens import HookedTransformer

import tqdm

from dataset import ModelDataset
from mlp import GeluMLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = HookedTransformer.from_pretrained_no_processing("pythia-70m-v0")
model = HookedTransformer.from_pretrained_no_processing("pythia-1b-v0")

# text_data = load_dataset("NeelNanda/pile-10k", split="train")
text_data = load_dataset("openwebtext", split="train[:10%]")
text_dataset = tokenize_and_concatenate(text_data, model.tokenizer)

layer_idx = 1
model_dataset = ModelDataset(model, layer_idx=layer_idx, dataset=text_dataset, batch_size=8, device=device)

mlp1x = GeluMLP(
    input_size=model.cfg.d_model,
    hidden_size=1*4*model.cfg.d_model,
    output_size=model.cfg.d_model,
    ).to(device)

mlp2x = GeluMLP(
    input_size=model.cfg.d_model,
    hidden_size=2*4*model.cfg.d_model,
    output_size=model.cfg.d_model,
    ).to(device)

mlp4x = GeluMLP(
    input_size=model.cfg.d_model,
    hidden_size=4*4*model.cfg.d_model,
    output_size=model.cfg.d_model,
    ).to(device)

num_steps = 40000
mlps = [mlp1x, mlp2x, mlp4x]
optimizers = [torch.optim.AdamW(mlp.parameters(), lr=1e-4) for mlp in mlps]
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps) for optimizer in optimizers]

writer = SummaryWriter()

for batch_idx, (pre, post) in tqdm.tqdm(enumerate(model_dataset), total=num_steps):
    # pre and post have shape (batch_size * seq_len, d_model)
    # add noise to input
    pre = pre + torch.randn_like(pre) * 0.1
    # post = post + torch.randn_like(post) * 0.001

    for mlp, optimizer, scheduler in zip(mlps, optimizers, schedulers):
        optimizer.zero_grad()
        y = mlp(pre)
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
