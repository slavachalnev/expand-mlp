from datasets import load_dataset
import torch

from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens import HookedTransformer

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

for idx, (pre, post) in enumerate(model_dataset):
    # pre and post have shape (batch_size * seq_len, d_model)
    print(f"Batch {idx}")

    for mlp, optimizer in zip(mlps, optimizers):
        optimizer.zero_grad()
        y = mlp(pre)
        loss = torch.nn.functional.mse_loss(y, post)
        loss.backward()
        optimizer.step()
        print(loss.item())

