import utils as u
import torch
import torch.distributed as dist
import numpy as np
import random
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

#datasets
import bitcoin_dl as bc
import elliptic_temporal_dl as ell_temp
import uc_irv_mess_dl as ucim
import auto_syst_dl as aus
import sbm_dl as sbm
import reddit_dl as rdt
from scipy.sparse import coo_matrix
# import test_dl as test

#taskers
import link_pred_tasker as lpt
import edge_cls_tasker as ect
import node_cls_tasker as nct

#models
import models as mls
import egcn_h
import egcn_o

import splitter as sp
import Cross_Entropy as ce

import trainer as tr

from run_exp import random_param_value, build_random_hyper_params, build_dataset, build_tasker, build_gcn, build_classifier

import logger
import os
import shutil
import networkx as nx

def save_gexf(df, filepath):
	network = nx.Graph()
	for i, edge in df.iterrows():
		if not network.has_edge(edge["source"], edge["target"]):
			network.add_edge(edge["source"], edge["target"], label=edge["label"])
	nx.write_gexf(network, filepath)

class Predictor():
	def __init__(self, args, splitter, gcn, classifier, dataset):
		self.args = args
		self.splitter = splitter
		self.tasker = splitter.tasker
		self.gcn = gcn
		self.classifier = classifier
		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.load_checkpoint(args.modelpath)

		# node_uids_path = os.path.join(args.fs_args.folder, args.fs_args.node_uids_file)
		# self.node_dict = pd.read_csv(node_uids_path)
		# self.node_dict = dict(zip(self.node_dict["encoding"].tolist(), self.node_dict["persistentid"].tolist()))
		# l1cat_uid_path = os.path.join(args.fs_args.folder, args.fs_args.l1_uids_file)
		# self.l1cat_dict = pd.read_csv(l1cat_uid_path)
		# self.l1cat_dict = dict(zip(self.l1cat_dict["encoding"].tolist(), self.l1cat_dict["level1cat"].tolist()))
		# path = os.path.join(self.args.fs_args.folder, self.args.fs_args.edges_file)
		# edges = pd.read_csv("alpha_0.35_beta_0.9_gamma_0.35_f_i_1_16_days.csv")
		# self.gt_df = pd.read_csv(path)

	def load_checkpoint(self, filename):
		if os.path.isfile(filename):
			checkpoint = torch.load(filename)
			epoch = checkpoint['epoch']
			self.gcn.GRCU_layers.load_state_dict(checkpoint['gcn_dict'])
			self.classifier.load_state_dict(checkpoint['classifier_dict'])
			return epoch
		else:
			self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
			return 0
	
	def run(self, split, start_time=0):
		raw_predictions_df, predictions_df = [], []
		torch.set_grad_enabled(False)
		for t, s in zip(np.arange(len(split)), split): 
			if t < start_time:
				continue
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				s = self.prepare_sample(s)
			predictions, nodes_embs = self.predict(s.hist_adj_list,
												   s.hist_ndFeats_list,
												   s.label_sp['idx'],
												   s.node_mask_list)
			probs = torch.softmax(predictions,dim=1)[:,1]
			probs = probs.cpu().numpy()         
			print("Probs Shape" +str(probs.shape))
			adj = s.label_sp['idx'].cpu().numpy()
			true = s.label_sp['vals'].cpu().numpy()           
			print("True Classes Shape: " + str(true.shape))
			true_matrix = coo_matrix((true,(adj[0],adj[1]))).toarray()
			true_edges_list = np.argwhere(true_matrix>0.5)
			true_edges_df = pd.DataFrame(true_edges_list)
			true_edges_df.to_csv("true_edges.csv",index = False)
			print(true_edges_df.shape)
			print("True Edges Shape:" + str(true_edges_list.shape))            
			pred_matrix = coo_matrix((probs,(adj[0],adj[1]))).toarray()
			# print(pred_matrix.shape)
			edges_list = np.argwhere(pred_matrix > 0.5)
			print(edges_list.shape)
			edges_df = pd.DataFrame(edges_list)
			edges_df.to_csv("predicted_edges.csv",index = False)
			mAP = self.get_MAP(predictions,s.label_sp['idx'], True)
			print("MAP: " + str(mAP))
			#print("Predicted Edges Shape 0.50:" +str(edges_list.shape))
			# edges_list = np.argwhere(pred_matrix > 0.6255)
			# # print(edges_list)
			# print("Predicted Edges Shape 0.625:" +str(edges_list.shape))
			# # edges = [adj[0]
			# print(pred_matrix)
			# predictions_df = None            
			# edges = s.label_sp["idx"].T.to("cpu").numpy()
			# preds = preds.to("cpu").numpy().reshape((-1,1))
			# predictions = np.concatenate((edges,preds), axis=1)
			# raw_predictions_df += ([[edge[0], edge[1], t, edge[2]] for edge in predictions])
			# predictions_df  += ([ [self.node_dict[edge[0]], self.node_dict[edge[1]], t, self.l1cat_dict[edge[2]]] for edge in predictions])
		# raw_predictions_df = pd.DataFrame(raw_predictions_df, columns=["source", "target", "time", "label"])
		# predictions_df = pd.DataFrame(predictions_df, columns=["source", "target", "time", "label"])

		# if self.args.out_args["save_csv"] or self.args.out_args["save_gexf"]:
		# 	if os.path.exists(self.args.out_args["folder"]):
		# 		shutil.rmtree(self.args.out_args["folder"])
		# 	os.makedirs(self.args.out_args["folder"])
		# 	if self.args.out_args["save_csv"]:
		# 		path = os.path.join(self.args.out_args["folder"], "predictions.csv")
		# 		predictions_df.to_csv(path, index=False)
		# 	if self.args.out_args["save_gexf"]:
		# 		folder = os.path.join(self.args.out_args["folder"], "gexf")
		# 		os.makedirs(folder)
		# 		for t, predictions in predictions_df.groupby("time"):
		# 			path = os.path.join(folder, str(t)+".gexf")
		# 			save_gexf(predictions, path)
		# if self.args.out_args["eval"]:
		# 	evaluations = {}
		# 	evaluations["overview"] = {}
		# 	class_numbers = self.gt_df["label"].value_counts()
		# 	class_numbers = dict(zip(class_numbers.keys(), class_numbers))
		# 	class_numbers = {self.l1cat_dict[cls]:cnt for cls, cnt in class_numbers.items()}
		# 	evaluations["overview"]["class_numbers"] = class_numbers
		# 	evaluations["MAP"], evaluations["MAP_by_class"] = self.get_MAP(raw_predictions_df)
		# 	print(evaluations)
		return predictions_df

	def get_MAP(self,predictions,true_classes, do_softmax=False):
		if do_softmax:
			probs = torch.softmax(predictions,dim=1)[:,1]
		else:
			probs = predictions

		predictions_np = probs.detach().cpu().numpy()
		true_classes_np = true_classes.detach().cpu().numpy()

		return average_precision_score(true_classes_np, predictions_np)    

	def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):
		nodes_embs = self.gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)
		predict_batch_size = 100000
		gather_predictions=[]
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	def prepare_sample(self,sample):
		sample = u.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
			sample.hist_adj_list[i] = adj.to(self.args.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

		label_sp = self.ignore_batch_dim(sample.label_sp)

		if self.args.task in ["link_pred", "edge_cls"]:
			label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
		else:
			label_sp['idx'] = label_sp['idx'].to(self.args.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp

		return sample

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []
		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)

	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj


if __name__ == '__main__':
	parser = u.create_parser()
	args = u.parse_args(parser)

	global rank, wsize, use_cuda
	args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
	args.device='cpu'
	if args.use_cuda:
		args.device='cuda'
	print ("use CUDA:", args.use_cuda, "- device:", args.device)
	try:
		dist.init_process_group(backend='mpi') #, world_size=4
		rank = dist.get_rank()
		wsize = dist.get_world_size()
		print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
		if args.use_cuda:
			torch.cuda.set_device(rank )  # are we sure of the rank+1????
			print('using the device {}'.format(torch.cuda.current_device()))
	except:
		rank = 0
		wsize = 1
		print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank, wsize)))
	
	if args.seed is None and args.seed!='None':
		seed = 123+rank#int(time.time())+rank
	else:
		seed=args.seed#+rank
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	args.seed=seed
	args.rank=rank
	args.wsize=wsize

	# Assign the requested random hyper parameters
	args = build_random_hyper_params(args)
	
	#build the dataset
	dataset = build_dataset(args)
	#build the tasker
	tasker = build_tasker(args, dataset)
	#build the splitter
	splitter = sp.splitter(args, tasker,pred=True)
	#build the models
	gcn = build_gcn(args, tasker)
	classifier = build_classifier(args, tasker)
	
	predictor = Predictor(args, splitter, gcn, classifier, dataset)
	predictor.run(splitter.test)