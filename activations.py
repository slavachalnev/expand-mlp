# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from ngram_ds import BigramFeatureDataset, FeatureDatasetConfig
from mlp import GeluMLP

from transformer_lens import HookedTransformer


dataset_config = FeatureDatasetConfig(
    dataset_name='NeelNanda/pile-10k',
    tokenizer_name='pythia',
    ctx_len=24,
    n_sequences=1000,
)
ngram_ds = BigramFeatureDataset('ngram').load(dataset_config)

model = HookedTransformer.from_pretrained_no_processing("pythia-1b-v0")
tokenizer = model.tokenizer
# ngram_ds[i] is {'tokens': tensor, 'label': "bigram" or "missing_first", or "missing_second", 'feature_name': "magnetic-field"}

label_to_idx = {
    'bigram': 0,
    'missing_first': 1,
    'missing_second': 2,
    }

mlps_dir = 'odin/mlps'
layer = 1
mlp_names = [f'mlp_8192_layer_{layer}.pt', f'mlp_16384_layer_{layer}.pt', f'mlp_32768_layer_{layer}.pt']
mlp_state_dicts = [torch.load(os.path.join(mlps_dir, mlp_name), map_location='cpu') for mlp_name in mlp_names]

mlps = []
for mlp_state_dict in mlp_state_dicts:
    mlp = GeluMLP(
        input_size=model.cfg.d_model,
        hidden_size=mlp_state_dict['fc1.weight'].shape[0],
        output_size=model.cfg.d_model,
        )
    mlp.load_state_dict(mlp_state_dict)
    mlps.append(mlp)


mlp_inputs = None
def inputs_hook(value, hook):
    h = value.detach().clone()
    global mlp_inputs
    mlp_inputs = h[:, -1, :]
    return value

mlp_hidden = None
def hidden_hook(value, hook):
    h = value.detach().clone()
    global mlp_hidden
    mlp_hidden = h[:, -1, :]
    return value

hooks = [
    (f'blocks.{layer}.ln2.hook_normalized', inputs_hook),
    (f'blocks.{layer}.mlp.hook_pre', hidden_hook),
    ]


act_sums = [] # for every mlp we have a tensor of shape (h_size, 3)
for h_size in [8192, 16384, 32768]:
    act_sums.append(torch.zeros((h_size, 3)))
counts = torch.zeros((3,))

with model.hooks(fwd_hooks=hooks), torch.no_grad():
    for i, item in enumerate(ngram_ds):
        label_idx = label_to_idx[item['label']]
        counts[label_idx] += 1

        model.forward(item['tokens'].unsqueeze(0), stop_at_layer=layer+1)

        for mlp_i, mlp in enumerate(mlps):
            h = mlp.forward(mlp_inputs, return_pre_act=True).squeeze(0)
            act_sums[mlp_i][:, label_idx] += h

act_means = [act_sum / counts for act_sum in act_sums]

# get top 10 neurons by largest difference between ngram types
top_neuron_idxs = []
for act_mean in act_means:
    diffs = act_mean[:, 0] - 0.5 * (act_mean[:, 1] + act_mean[:, 2])
    top_neuron_idxs.append(diffs.argsort(descending=True)[:10])  # get top 10 neurons
    print(diffs[top_neuron_idxs[-1]])


# record neuron activations for top neurons

# Prepare empty lists to store neuron activations
neuron_activations = [[[[] for _ in range(3)] for _ in range(5)] for _ in range(len(top_neuron_idxs))]

with model.hooks(fwd_hooks=hooks), torch.no_grad():
    for i, item in enumerate(ngram_ds):
        label_idx = label_to_idx[item['label']]

        model.forward(item['tokens'].unsqueeze(0), stop_at_layer=layer+1)

        for mlp_i, mlp in enumerate(mlps):
            h = mlp.forward(mlp_inputs, return_pre_act=True).squeeze(0)
            for idx, neuron_idx in enumerate(top_neuron_idxs[mlp_i][:5]):  # only take the top 5 neurons
                neuron_activations[mlp_i][idx][label_idx].append(h[neuron_idx].item())  # save the activation of top neurons

# %%

# Compute min and max across all activations
overall_min = min([min([min(act) for act in neuron]) for mlp in neuron_activations for neuron in mlp])
overall_max = max([max([max(act) for act in neuron]) for mlp in neuron_activations for neuron in mlp])

# Compute bins
n_bins = 30
bins = np.linspace(overall_min, overall_max, n_bins)

# Plotting the histograms
fig, axs = plt.subplots(len(mlps)*5, 1, figsize=(10, 5 * len(mlps) * 5))
labels = ['bigram', 'missing_first', 'missing_second']

for i, neuron_label_activations in enumerate(neuron_activations):
    for j, label_activations in enumerate(neuron_label_activations):
        for k, activations in enumerate(label_activations):
            axs[i*5+j].hist(activations, bins=bins, alpha=0.5, label=labels[k])
        axs[i*5+j].set_title(f'Activations of Top Neuron {j+1} for MLP {i+1}')
        axs[i*5+j].legend()

plt.tight_layout()
plt.savefig('activations.png')
plt.show()

# %%
