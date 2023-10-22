from typing import Any

import argparse
import torch

from dataclasses import dataclass


@dataclass
class BaseArgs:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        for key, value in vars(self).items():
            parser.add_argument(f"--{key}", type=type(value), default=None)
        return parser.parse_args()
        
    def __post_init__(self) -> None:
        # parse command line arguments and update the class
        command_line_args = self.parse_args()
        extra_args = set(vars(command_line_args)) - set(vars(self))
        if extra_args: 
            raise ValueError(f"Unknown arguments: {extra_args}")
        self.update(command_line_args)
    
    def update(self, args: Any) -> None:
        for key, value in vars(args).items():
            if value is not None:
                print(f"From command line, setting {key} to {value}")
                setattr(self, key, value)


@dataclass
class InterpArgs(BaseArgs):
    layer: int = 2
    model_name: str = "EleutherAI/pythia-70m-deduped"
    layer_loc: str = "residual"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_feats_explain: int = 10
    # load_interpret_autoencoder: str = ""
    tied_ae: bool = False
    interp_name: str = ""
    sort_mode: str = "max"
    use_decoder: bool = True
    df_n_feats: int = 200
    top_k: int = 50
    save_loc: str = ""

