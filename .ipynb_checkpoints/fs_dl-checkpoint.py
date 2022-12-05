import torch
import pandas as pd
import utils as u
import os
import sklearn

class Foursquare_Dataset():
    def __init__(self,args):
        args.fs_args = u.Namespace(args.fs_args)
        folder = args.fs_args.folder

        nodefile = args.fs_args.nodes_file
        nodefile = os.path.join(folder, nodefile)
        node_feats = pd.read_csv(nodefile)
        node_feats = torch.tensor(node_feats.values[:,1:], dtype=torch.long)
        num_nodes = node_feats.size(0)

        edgefile = args.fs_args.edges_file
        edgefile = os.path.join(folder, edgefile)
        edges = pd.read_csv(edgefile)
        num_classes = len(edges["label"].unique())
        class_weights = sklearn.utils.class_weight.compute_class_weight("balanced", classes=edges["label"].unique(), y=edges["label"].to_numpy())
        class_weights = {cls:weight for cls, weight in zip(edges["label"].unique(), class_weights)}
        class_weights = [val for key, val in sorted(class_weights.items(), key=lambda x: x[0])]
        class_weights = torch.tensor(class_weights)
        edges = torch.tensor(edges.values, dtype=torch.long)

        self.min_time = edges[:,2].min().type(torch.int)
        self.max_time = edges[:,2].max().type(torch.int)
        self.nodes_feats = node_feats
        self.num_nodes = num_nodes
        self.feats_per_node = node_feats.size(1)
        self.num_classes = num_classes
        self.edges = {'idx': edges, 'vals': edges[:,3]}
        self.class_weights = class_weights

        print("#nodes = ", num_nodes, "; #edges = ", edges.shape[0])