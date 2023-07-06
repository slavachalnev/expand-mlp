from datasets import load_dataset
import torch

from torch.utils.tensorboard import SummaryWriter

from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens import HookedTransformer

import tqdm

from dataset import ModelDataset
from mlp import GeluMLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained_no_processing("pythia-70m-v0")

text_data = load_dataset("NeelNanda/pile-10k", split="train")
text_dataset = tokenize_and_concatenate(text_data, model.tokenizer)

model_dataset = ModelDataset(model, layer_idx=1, dataset=text_dataset, batch_size=8, device=device)

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

mlps = [mlp1x, mlp2x, mlp4x]
optimizers = [torch.optim.AdamW(mlp.parameters(), lr=1e-3) for mlp in mlps]

writer = SummaryWriter()

num_steps = 10000
for batch_idx, (pre, post) in tqdm.tqdm(enumerate(model_dataset), total=num_steps):
    # pre and post have shape (batch_size * seq_len, d_model)
    for mlp, optimizer in zip(mlps, optimizers):
        optimizer.zero_grad()
        y = mlp(pre)
        loss = torch.nn.functional.mse_loss(y, post)
        loss.backward()
        optimizer.step()
        writer.add_scalar(f"loss/{mlp.hidden_size}", loss.item(), batch_idx)
    
    if batch_idx > num_steps:
        print("Done")
        break

# save
for mlp in mlps:
    torch.save(mlp.state_dict(), f"mlps/mlp_{mlp.hidden_size}.pt")

writer.close()
