import numpy as np
import itertools
import torch.nn as nn
import torch
import torch.optim as optim
import os
import pandas as pd
from similarity_search_class import load_embedding

from similarity_search_class import get_accepted_frames_bool_list

base_analysis_dir = "analysis"
base_labeled_df_dir = os.path.join(base_analysis_dir, "ultra/labeled_dfs")
base_output_dir = "output"
base_results_dir = os.path.join(base_analysis_dir, "results")
base_embeddings_dir = os.path.join(base_output_dir, "embeddings")

# 30 frames
results_pq_path = os.path.join(
    base_results_dir,
    f"similarity_variable,{'bdd'},clc@0.1,{'redcars'}-otherstuff_tiling",
)

results_df = pd.read_parquet(results_pq_path)
full_dataset = pd.read_parquet(
    os.path.join(
        base_results_dir,
        f"results_variable,{'bdd'},clc@0.1,{'redcars'}-otherstuff",
    )
)


true_false_list = get_accepted_frames_bool_list(
    [2, 3, 4, 15, 20, 21, 27, 29], 30
)
full_dataset.loc[:, "embedding_array"] = full_dataset["clip_embedding"].apply(
    lambda x: load_embedding(x)
)


def label_df(df, true_false_list):
    df["svm_label"] = true_false_list
    df.loc[:, "embedding_array"] = df["clip_embedding"].apply(
        lambda x: load_embedding(x)
    )
    return df


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.sequential = nn.Sequential(
            self.fc1, self.relu, self.fc2, self.sigmoid
        )

    def forward(self, x):
        return self.sequential(x)


def compute_loss(scores_pair):
    loss = 0
    for pair in scores_pair:
        loss += torch.abs(pair[0] - pair[1]) + min(
            torch.abs(pair[0] - pair[2]), torch.abs(pair[1] - pair[2])
        )
    return loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # input to model: 30 x 512
    model = MLP(512, 256, 1)
    # model = model.cuda()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # input to loss fn: (? x 3) where [p1, p2, n]
    model.train()
    df = label_df(results_df, true_false_list)
    embeddings = df["embedding_array"].to_numpy()
    embeddings = torch.from_numpy(np.stack((embeddings)))
    scores = model(embeddings)

    true_false_np = np.array(true_false_list)
    pos_indices = np.where(true_false_np == 1)[0]
    neg_indices = np.where(true_false_np == 0)[0]
    out = itertools.product(pos_indices, pos_indices, neg_indices)
    valid_pairs = []
    for tup in out:
        if tup[0] != tup[1]:
            valid_pairs.append(list(tup))

    scores = scores.flatten()
    scores_pair = []
    for pair in valid_pairs:
        scores_i = scores[pair]
        scores_pair.append(scores_i)
    scores_pair = torch.stack(scores_pair)
    loss = compute_loss(scores_pair)
    loss.backward()
    optimizer.step()
    embeddings_full = full_dataset["embedding_array"].to_numpy()
    # e_full = torch.from_numpy(np.stack((embeddings_full))).cuda()
    e_full = torch.from_numpy(np.stack((embeddings_full)))
    scores_full = model(e_full)
    print(scores_full)
    # prods = list(zip(itertools.product(pos_indices, pos_indices, neg_indices)))


main()
