import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

from ngram_ds import BigramFeatureDataset, FeatureDatasetConfig
from mlp import GeluMLP, SoluMLP

from transformer_lens import HookedTransformer


def analyse_feature(
        feature_name: str,
        mlp_type: str = 'gelu', # 'gelu' or 'solu'
        mlp_dir: str = 'mlps',
        layer=1,
        n_sequences=8000,
        # n_pre_sort=100,
        dataset_name='NeelNanda/pile-10k',
        mlp_dims=None,
    ):
    if mlp_dims is None:
        mlp_dims = [8192, 16384, 32768]

    dataset_config = FeatureDatasetConfig(
        dataset_name=dataset_name,
        tokenizer_name='pythia',
        ctx_len=24,
        n_sequences=n_sequences,
    )
    ngram_ds = BigramFeatureDataset('ngram').load(dataset_config)
    filtered_dataset = list(filter(lambda x: x['feature_name'] == feature_name, ngram_ds))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HookedTransformer.from_pretrained_no_processing("pythia-1b-v0")
    # ngram_ds[i] is {'tokens': tensor, 'label': "bigram" or "missing_first", or "missing_second", 'feature_name': "magnetic-field"}

    label_to_idx = {
        'bigram': 0,
        'missing_first': 1,
        'missing_second': 2,
        }

    mlp_names = [f'mlp_{dim}_layer_{layer}.pt' for dim in mlp_dims]
    mlp_state_dicts = [torch.load(os.path.join(mlp_dir, mlp_name), map_location='cpu') for mlp_name in mlp_names]
    mlp_class = GeluMLP if mlp_type == 'gelu' else SoluMLP

    mlps = []
    for mlp_state_dict in mlp_state_dicts:
        mlp = mlp_class(
            input_size=model.cfg.d_model,
            hidden_size=mlp_state_dict['fc1.weight'].shape[0],
            output_size=model.cfg.d_model,
            )
        mlp.load_state_dict(mlp_state_dict)
        mlp.to(device)
        mlps.append(mlp)


    mlp_state = dict()
    def inputs_hook(value, hook):
        h = value.detach().clone().cpu()
        mlp_state['input'] = h[:, -1, :]
        return value

    def hidden_hook(value, hook):
        h = value.detach().clone().cpu()
        mlp_state['hidden'] = h[:, -1, :]
        return value

    hooks = [
        (f'blocks.{layer}.ln2.hook_normalized', inputs_hook),
        # (f'blocks.{layer}.mlp.hook_pre', hidden_hook),
        (f'blocks.{layer}.mlp.hook_post', hidden_hook),
        ]


    # act_sums = [] # for every mlp we have a tensor of shape (h_size, 3)
    # for h_size in mlp_dims + [model.cfg.d_model * 4]:  # last one is for the original model.
    #     act_sums.append(torch.zeros((h_size, 3)))
    # counts = torch.zeros((3,))

    # with model.hooks(fwd_hooks=hooks), torch.no_grad():
    #     for i, item in tqdm(enumerate(filtered_dataset)):
    #         label_idx = label_to_idx[item['label']]
    #         counts[label_idx] += 1

    #         model.forward(item['tokens'].unsqueeze(0), stop_at_layer=layer+1)

    #         for mlp_i, mlp in enumerate(mlps):
    #             h = mlp.forward(mlp_state['input'].to(device), return_post_act=True).squeeze(0).cpu()
    #             act_sums[mlp_i][:, label_idx] += h
    #         act_sums[-1][:, label_idx] += mlp_state['hidden'].squeeze(0).cpu() # for the original model

    # act_means = [act_sum / counts for act_sum in act_sums]

    # get top n_pre_sort neurons by largest difference between ngram types
    # top_neuron_idxs = []
    # for act_mean in act_means:
        # diffs = act_mean[:, 0] - torch.max(act_mean[:, 1], act_mean[:, 2])
        # sorted_idx = diffs.argsort(descending=True)
        # top_neuron_idxs.append(sorted_idx[:n_pre_sort].tolist())
        # # print(diffs[top_neuron_idxs[-1]])
    
    top_neuron_idxs = []
    # hack
    for mlp in mlps:
        # append all neurons
        top_neuron_idxs.append(list(range(mlp.hidden_size)))
    top_neuron_idxs.append(list(range(model.cfg.d_model * 4))) # for the original model


    # Prepare empty lists to store neuron activations
    neuron_activations = [[[[] for _ in range(3)] for _ in range(mlp.hidden_size)] for mlp in mlps]
    neuron_activations.append([[[] for _ in range(3)] for _ in range(model.cfg.d_model * 4)]) # for the original model

    with model.hooks(fwd_hooks=hooks), torch.no_grad():
        for i, item in tqdm(enumerate(filtered_dataset)):
            label_idx = label_to_idx[item['label']]

            model.forward(item['tokens'].unsqueeze(0), stop_at_layer=layer+1)

            for mlp_i, mlp in enumerate(mlps):
                # h = mlp.forward(mlp_state['input'].to(device), return_pre_act=True).squeeze(0).cpu()
                h = mlp.forward(mlp_state['input'].to(device), return_post_act=True).squeeze(0).cpu()
                for idx, neuron_idx in enumerate(top_neuron_idxs[mlp_i]):
                    neuron_activations[mlp_i][idx][label_idx].append(h[neuron_idx].item())  # save the activation of top neurons
            for idx, neuron_idx in enumerate(top_neuron_idxs[-1]):  # for the original model
                neuron_activations[-1][idx][label_idx].append(mlp_state['hidden'].squeeze(0).cpu()[neuron_idx].item())
    
    return neuron_activations, top_neuron_idxs


def rank_by_classifier(neuron_activations, top_neuron_idxs):
    n_mlps = len(neuron_activations)

    # Training a classifier for each neuron and recording the accuracy, precision, and recall
    classifier_metrics = [[{"accuracy": 0, "precision": 0, "recall": 0} for _ in top_neuron_idxs[mlp_i]] for mlp_i in range(n_mlps)]

    for mlp_i in range(n_mlps):
        for neuron_i in range(len(top_neuron_idxs[mlp_i])):
            # Preparing the data
            X = neuron_activations[mlp_i][neuron_i][0] + \
                neuron_activations[mlp_i][neuron_i][1] + \
                neuron_activations[mlp_i][neuron_i][2]
            y = [0]*len(neuron_activations[mlp_i][neuron_i][0]) + \
                [1]*(len(neuron_activations[mlp_i][neuron_i][1]) + len(neuron_activations[mlp_i][neuron_i][2]))

            # Training the classifier
            clf = LogisticRegression(random_state=42, penalty=None).fit(np.array(X).reshape(-1, 1), y)

            y_pred = clf.predict(np.array(X).reshape(-1, 1))

            # Recording the accuracy, precision and recall
            classifier_metrics[mlp_i][neuron_i]["accuracy"] = clf.score(np.array(X).reshape(-1, 1), y)
            classifier_metrics[mlp_i][neuron_i]["precision"] = precision_score(y, y_pred)
            classifier_metrics[mlp_i][neuron_i]["recall"] = recall_score(y, y_pred)

    top_neuron_idxs_ranked = [
        sorted(
            range(len(metrics)), key=lambda i: -metrics[i]["accuracy"]
        )[:5] 
        for metrics in classifier_metrics
    ]  # get top 5 neurons
    
    classifier_metrics_ranked = [
        [
            classifier_metrics[mlp_i][neuron_i] 
            for neuron_i in top_neuron_idxs_ranked[mlp_i]
        ] 
        for mlp_i in range(n_mlps)
    ]  # get metrics for top 5 neurons

    neuron_activations_ranked = []
    for mlp_i in range(n_mlps):
        neuron_activations_ranked.append([])
        for neuron_i in top_neuron_idxs_ranked[mlp_i]:
            neuron_activations_ranked[mlp_i].append(neuron_activations[mlp_i][neuron_i])
    neuron_activations = neuron_activations_ranked

    return neuron_activations, top_neuron_idxs_ranked, classifier_metrics_ranked


def plot_hist(neuron_activations, top_neuron_idxs, feature_name, mlp_dir='mlps', mlp_dims=None):
    if mlp_dims is None:
        mlp_dims = [8192, 16384, 32768]
    n_mlps = len(neuron_activations)
    n_neurons = len(neuron_activations[0])

    n_bins = 100

    fig, axs = plt.subplots(n_neurons, n_mlps, figsize=(5*n_mlps, 5*n_neurons))
    labels = ['bigram', 'missing_first', 'missing_second']
    mlp_names = [str(dim) for dim in mlp_dims] + ['original']

    for i, neuron_label_activations in enumerate(neuron_activations):
        print('len neuron_label_activations', len(neuron_label_activations))
        for j, label_activations in enumerate(neuron_label_activations):
            neuron_idx = top_neuron_idxs[i][j] # Get the actual neuron index
            for k, activations in enumerate(label_activations):

                bins = np.linspace(min(activations), max(activations), n_bins)

                axs[j, i].hist(activations, bins=bins, alpha=0.5, label=labels[k])
                axs[j, i].set_yscale('log')
            axs[j, i].set_title(f'Activations of Neuron {neuron_idx} for mlp_{mlp_names[i]}')
            axs[j, i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(mlp_dir, f'{feature_name}_activations.png'))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-dir", type=str, required=True, help="Dir to load MLPs and save plots.")
    parser.add_argument("--mlp-type", default='gelu', type=str, help="Type of MLP to use. gelu or solu.")
    parser.add_argument('--mlp-dims', nargs='+', type=int, default=None, help='List of MLP dimensions')
    args = parser.parse_args()

    feature_names = [ # layer 1
        'magnetic-field',
        'human-rights',
        'north-america',
        'gene-expression',
        'mental-health',
        'side-effects',
        ]
    
    for feature_name in feature_names:
        print(f'Analysing {feature_name}')
        neuron_activations, top_neuron_idxs = analyse_feature(feature_name,
                                                              mlp_type=args.mlp_type,
                                                              mlp_dir=args.mlp_dir,
                                                              dataset_name='openwebtext',
                                                              mlp_dims=args.mlp_dims,
                                                              )
        neuron_activations, top_neuron_idxs, classifier_metrics = rank_by_classifier(neuron_activations, top_neuron_idxs)
        print(classifier_metrics)
        plot_hist(neuron_activations, top_neuron_idxs, feature_name, mlp_dir=args.mlp_dir, mlp_dims=args.mlp_dims)
