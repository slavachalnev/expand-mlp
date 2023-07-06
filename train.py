from datasets import load_dataset
import torch

from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens import HookedTransformer

from dataset import ModelDataset
from mlp import GeluMLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("pythia-70m-v0")

text_data = load_dataset("NeelNanda/pile-10k", split="train")
text_dataset = tokenize_and_concatenate(text_data, model.tokenizer)

model_dataset = ModelDataset(model, layer_idx=1, dataset=text_dataset, batch_size=32, device=device)

mlp2x = GeluMLP(
    input_size=model.cfg.d_model,
    hidden_size=2*4*model.cfg.d_model,
    output_size=model.cfg.d_model,
    ).to(device)


for pre_h, post_h in model_dataset:
    print(pre_h.shape)
    print(post_h.shape)
    break


