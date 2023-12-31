from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

from transformer_lens.utils import tokenize_and_concatenate


class ModelDataset:
    """Dataset of pre and post MLP activations"""

    def __init__(self,
                 model,
                 layer_idx,
                 dataset,
                 batch_size,
                 device,
                 skip_connection=False,
                 parallel=True,  # parallel attn (e.g. pythia)
                 ):
        
        if not parallel:
            raise NotImplementedError("Only parallel attn is supported")
        if skip_connection:
            raise NotImplementedError("Skip connection not supported")

        self.model = model # HookedTransformer
        self.d_model = model.cfg.d_model
        self.layer_idx = layer_idx

        self.data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.device = device

        self.pre_h = None
        self.post_h = None

        def pre_hook(value, hook):
            h = value.detach().clone()
            h = h.reshape(-1, self.d_model)
            self.pre_h = h
            return value
        
        def post_hook(value, hook):
            h = value.detach().clone()
            h = h.reshape(-1, self.d_model)
            self.post_h = h
            return value

        self.fwd_hooks = [
            (f"blocks.{layer_idx}.ln2.hook_normalized", pre_hook),
            (f"blocks.{layer_idx}.hook_mlp_out", post_hook),
            ]

    @torch.no_grad()
    def run_model(self, batch):
        toks = batch["tokens"].to(self.device)
        with self.model.hooks(fwd_hooks=self.fwd_hooks):
            self.model(toks, stop_at_layer=self.layer_idx+1)
    
    def __iter__(self):
        while True:
            for batch in self.data_loader:
                self.run_model(batch)
                yield self.pre_h, self.post_h
        