import os, sys

import shutil
import time

import math

import csv
import random

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


import torch
from torch import nn
import torch.nn.functional as F 
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


import tokenizers
from tokenizers import Tokenizer
from transformers import BertTokenizerFast, BertModel

import os.path as osp
import zipfile

import torch
from torch_geometric.data import download_url, Data
from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

#Need a special generator for random sampling:
import argparse

parser = argparse.ArgumentParser(description='Run attention weights recalculation')

parser.add_argument('--checkpoint', metavar='checkpoint', type=str, 
                    help='directory where checkpoint is located')


args = parser.parse_args()  
CHECKPOINT = args.checkpoint #'checkpoint/attention_checkpoint.pt'

class GenerateData():
  def __init__(self, path_train, path_val, path_test, path_molecules, path_token_embs):
    self.path_train = path_train
    self.path_val = path_val
    self.path_test = path_test
    self.path_molecules = path_molecules
    self.path_token_embs = path_token_embs

    self.mol_trunc_length = 512
    self.text_trunc_length = 256

    self.prep_text_tokenizer()

    self.load_substructures()

    self.batch_size = 32

    self.store_descriptions()

  def load_substructures(self):
    self.molecule_sentences = {}
    self.molecule_tokens = {}

    total_tokens = set()
    self.max_mol_length = 0
    with open(self.path_molecules) as f:
      for line in f:
        spl = line.split(":")
        cid = spl[0]
        tokens = spl[1].strip()
        self.molecule_sentences[cid] = tokens
        t = tokens.split()
        total_tokens.update(t)
        size = len(t)
        if size > self.max_mol_length: self.max_mol_length = size


    self.token_embs = np.load(self.path_token_embs, allow_pickle = True)[()]



  def prep_text_tokenizer(self):
    self.text_tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")


  def store_descriptions(self):
    self.descriptions = {}

    self.mols = {}



    self.training_cids = []
    #get training set cids...
    with open(self.path_train) as f:
      reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
      for n, line in enumerate(reader):
        self.descriptions[line['cid']] = line['desc']
        self.mols[line['cid']] = line['mol2vec']
        self.training_cids.append(line['cid'])

    self.training_cids_sample  = random.sample(self.training_cids, int(len(self.training_cids)/10))

    self.validation_cids = []
    #get validation set cids...
    with open(self.path_val) as f:
      reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
      for n, line in enumerate(reader):
        self.descriptions[line['cid']] = line['desc']
        self.mols[line['cid']] = line['mol2vec']
        self.validation_cids.append(line['cid'])
    self.validation_cids  = random.sample(self.validation_cids, int(len(self.validation_cids)/10))

    self.test_cids = []
    with open(self.path_test) as f:
      reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
      for n, line in enumerate(reader):
        self.descriptions[line['cid']] = line['desc']
        self.mols[line['cid']] = line['mol2vec']
        self.test_cids.append(line['cid'])
    self.test_cids  = random.sample(self.test_cids, int(len(self.test_cids)/10))

  #transformers can't take array with full attention so have to pad a 0...
  def padarray(self, A, size, value=0):
      t = size - len(A)
      return np.pad(A, pad_width=(0, t), mode='constant', constant_values = value)


  def generate_examples_train(self):
    """Yields examples."""

    np.random.shuffle(self.training_cids)

    for cid in self.training_cids:
      label = np.random.randint(2)
      rand_cid = np.random.choice(self.training_cids)
      if label:
        text_input = self.text_tokenizer(self.descriptions[cid], truncation=True, max_length=self.text_trunc_length - 1,
                                        padding='max_length', return_tensors = 'np')
      else:
        text_input = self.text_tokenizer(self.descriptions[rand_cid], truncation=True, max_length=self.text_trunc_length - 1,
                                        padding='max_length', return_tensors = 'np')

      text_ids = self.padarray(text_input['input_ids'].squeeze(), self.text_trunc_length)
      text_mask = self.padarray(text_input['attention_mask'].squeeze(), self.text_trunc_length)

      yield {
          'cid': cid,
          'input': {
              'text': {
                'input_ids': text_ids,
                'attention_mask': text_mask,
              },
              'molecule' : {
                    'mol2vec' : np.fromstring(self.mols[cid], sep = " "),
                    'cid' : cid
              },
          },
          'label': label
      }


  def generate_examples_val(self):
    """Yields examples."""

    np.random.shuffle(self.validation_cids)

    for cid in self.validation_cids:
      label = np.random.randint(2)
      rand_cid = np.random.choice(self.validation_cids)
      if label:
        text_input = self.text_tokenizer(self.descriptions[cid], truncation=True, max_length=self.text_trunc_length - 1,
                                        padding='max_length', return_tensors = 'np')
      else:
        text_input = self.text_tokenizer(self.descriptions[rand_cid], truncation=True, max_length=self.text_trunc_length - 1,
                                        padding='max_length', return_tensors = 'np')


      text_ids = self.padarray(text_input['input_ids'].squeeze(), self.text_trunc_length)
      text_mask = self.padarray(text_input['attention_mask'].squeeze(), self.text_trunc_length)

      yield {
          'cid': cid,
          'input': {
              'text': {
                'input_ids': text_ids,
                'attention_mask': text_mask,
              },
              'molecule' : {
                    'mol2vec' : np.fromstring(self.mols[cid], sep = " "),
                    'cid' : cid
              },
          },
          'label': label
      }

  def generate_examples_test(self):
    """Yields examples."""

    np.random.shuffle(self.test_cids)

    for cid in self.test_cids:
      label = np.random.randint(2)
      rand_cid = np.random.choice(self.test_cids)
      if label:
        text_input = self.text_tokenizer(self.descriptions[cid], truncation=True, max_length=self.text_trunc_length - 1,
                                        padding='max_length', return_tensors = 'np')
      else:
        text_input = self.text_tokenizer(self.descriptions[rand_cid], truncation=True, max_length=self.text_trunc_length - 1,
                                        padding='max_length', return_tensors = 'np')


      text_ids = self.padarray(text_input['input_ids'].squeeze(), self.text_trunc_length)
      text_mask = self.padarray(text_input['attention_mask'].squeeze(), self.text_trunc_length)

      yield {
          'cid': cid,
          'input': {
              'text': {
                'input_ids': text_ids,
                'attention_mask': text_mask,
              },
              'molecule' : {
                    'mol2vec' : np.fromstring(self.mols[cid], sep = " "),
                    'cid' : cid
              },
          },
          'label': label
      }


mounted_path_token_embs = "data/token_embedding_dict.npy"
mounted_path_train = "data/training.txt"
mounted_path_val = "data/val.txt"
mounted_path_test = "data/test.txt"
mounted_path_molecules = "data/ChEBI_definitions_substructure_corpus.cp"
gt = GenerateData(mounted_path_train, mounted_path_val, mounted_path_test, mounted_path_molecules, mounted_path_token_embs)

class Dataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, gen, length):
      'Initialization'

      self.gen = gen
      self.it = iter(self.gen())

      self.length = length

  def __len__(self):
      'Denotes the total number of samples'
      return self.length


  def __getitem__(self, index):
      'Generates one sample of data'

      try:
        ex = next(self.it)
      except StopIteration:
        self.it = iter(self.gen())
        ex = next(self.it)

      X = ex['input']
      y = ex['label']

      return X, y

training_set = Dataset(gt.generate_examples_train, len(gt.training_cids))
validation_set = Dataset(gt.generate_examples_val, len(gt.validation_cids))
test_set = Dataset(gt.generate_examples_test, len(gt.test_cids))

# Parameters
params = {'batch_size': gt.batch_size,
          'shuffle': True,
          'num_workers': 1}

training_generator = DataLoader(training_set, **params)
validation_generator = DataLoader(validation_set, **params)
test_generator = DataLoader(test_set, **params)

class MoleculeGraphDataset(GeoDataset):
    def __init__(self, root, cids, data_path, gt, transform=None, pre_transform=None):
        self.cids = cids
        self.data_path = data_path
        self.gt = gt
        super(MoleculeGraphDataset, self).__init__(root, transform, pre_transform)

        self.idx_to_cid = {}
        i = 0
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            self.idx_to_cid[i] = cid
            i += 1

    @property
    def raw_file_names(self):
        return [cid + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]

    def download(self):
        # Download to `self.raw_dir`.
        print('raw_dir_1', self.raw_dir)
        print(osp.join(self.raw_dir, "mol_graphs.zip"))
        print(osp.exists(osp.join(self.raw_dir, "mol_graphs.zip")))
        if not osp.exists(osp.join(self.raw_dir, "mol_graphs.zip")):
            shutil.copy(self.data_path, os.path.join(self.raw_dir, "mol_graphs.zip"))

    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: #edges
          if line != "\n":
            edge = *map(int, line.split()),
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f: #get mol2vec features:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.token_embs:
            x.append(self.gt.token_embs[substruct_id])
          else:
            x.append(self.gt.token_embs['UNK'])

        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)



    def process(self):
        print('raw_dir', self.raw_dir)
        with zipfile.ZipFile(os.path.join(self.raw_dir, "mol_graphs.zip"), 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)


        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.

            cid = int(raw_path.split('/')[-1][:-6])

            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index = edge_index)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data

#To get specific lists...

class CustomGraphCollater(object):
    def __init__(self, dataset, mask_len, follow_batch = [], exclude_keys = []):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.dataset = dataset
        self.mask_len = mask_len
        self.mask_indices = np.array(range(mask_len))

    def generate_mask(self, sz):
        rv = torch.zeros((self.mask_len), dtype = torch.bool)
        rv = rv.masked_fill(torch.BoolTensor(self.mask_indices < sz), bool(1)) #pytorch transformer input version
        rv[-1] = 0 #set last value to 0 because pytorch can't handle all 1s
        return rv

    def get_masks(self, batch):
      return torch.stack([self.generate_mask(b.x.shape[0]) for b in batch])

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch)

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, cids):

        tmp = [self.dataset.get_cid(int(cid)) for cid in cids]
        return self.collate(tmp), self.get_masks(tmp)

data_path = "data"
graph_data_path = osp.join(data_path, "mol_graphs.zip")
root = osp.join(graph_data_path[:-len(osp.basename(graph_data_path))], 'graph-data/')
#graph_data_path = "input/mol_graphs.zip"
print('root', root)
if not os.path.exists(root):
    os.mkdir(root)


mg_data_tr = MoleculeGraphDataset(root, gt.training_cids, graph_data_path, gt)
graph_batcher_tr = CustomGraphCollater(mg_data_tr, gt.mol_trunc_length)

mg_data_val = MoleculeGraphDataset(root, gt.validation_cids, graph_data_path, gt)
graph_batcher_val = CustomGraphCollater(mg_data_val, gt.mol_trunc_length)

mg_data_test = MoleculeGraphDataset(root, gt.test_cids, graph_data_path, gt)
graph_batcher_test = CustomGraphCollater(mg_data_test, gt.mol_trunc_length)

class Model(nn.Module):

    def __init__(self, ntoken, ninp, nout, nhid, nhead, nlayers, graph_hidden_channels, mol_trunc_length,  dropout=0.5):
        super(Model, self).__init__()

        self.text_hidden1 = nn.Linear(ninp, nhid)
        self.text_hidden2 = nn.Linear(nhid, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.graph_hidden_channels = graph_hidden_channels

        self.drop = nn.Dropout(p=dropout)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.text_transformer_decoder = TransformerDecoder(decoder_layers, nlayers)


        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()

        #For GCN:
        self.conv1 = GCNConv(mg_data_val.num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)


        self.other_params = list(self.parameters()) #get all but bert params

        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

        self.device = 'cpu'

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, graph_batch, text_mask = None, molecule_mask = None):

        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        #Obtain node embeddings
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        mol_x = self.conv3(x, edge_index)

        #turn pytorch geometric output into the correct format for transformer
        #requires recovering the nodes from each graph into a separate dimension
        node_features = torch.zeros((graph_batch.num_graphs, gt.mol_trunc_length, self.graph_hidden_channels)).to(self.device)
        for i, p in enumerate(graph_batch.ptr):
          if p == 0:
            old_p = p
            continue
          node_features[i - 1, :p-old_p, :] = mol_x[old_p:torch.min(p, old_p + gt.mol_trunc_length), :]
          old_p = p
        node_features = torch.transpose(node_features, 0, 1)

        text_output = self.text_transformer_decoder(text_encoder_output['last_hidden_state'].transpose(0,1), node_features,
                                                            tgt_key_padding_mask = text_mask == 0, memory_key_padding_mask = ~molecule_mask)


        #Readout layer
        x = global_mean_pool(mol_x, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x)
        x = x.relu()
        x = self.mol_hidden2(x)

        text_x = torch.tanh(self.text_hidden1(text_output[0,:,:])) #[CLS] pooler
        text_x = self.text_hidden2(text_x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x

mol_trunc_length=512
ntoken = gt.text_tokenizer.vocab_size
model = Model(ntoken = ntoken, ninp = 768, nout = 300, nhead = 8, nhid = 512, nlayers = 3, graph_hidden_channels = 768, mol_trunc_length=mol_trunc_length)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(CHECKPOINT, map_location=device), strict=False)

model.eval()
model.set_device(device)

last_decoder = model.text_transformer_decoder.layers[-1]

mha_weights = {}
def get_activation(name):
    def hook(model, input, output):
        if output[0] is not None:
        #     # print(output[0], output[1])
            mha_weights[cid] = output[0].cpu().detach().numpy()
        else:
            print("Attention weights are None for cid:", cid)
    return hook


handle = last_decoder.multihead_attn.register_forward_hook(get_activation(''))

for i,d in enumerate(gt.generate_examples_train()):

  batch = d['input']

  cid = d['cid']#batch['molecule']['cid'][0]
  text_mask = torch.Tensor(batch['text']['attention_mask']).bool().reshape(1,-1).to(device)

  text = torch.Tensor(batch['text']['input_ids']).int().reshape(1,-1).to(device)
  graph_batch, molecule_mask = graph_batcher_val([batch['molecule']['cid']])
  graph_batch = graph_batch.to(device)
  molecule_mask = molecule_mask.to(device)
  graph_batch.edge_index = graph_batch.edge_index.reshape((2,-1))

  out = model(text, graph_batch, text_mask, molecule_mask)

  #for memory reasons
  mol_length = graph_batch.x.shape[0]
  text_input = gt.text_tokenizer(gt.descriptions[cid], truncation=True, padding = 'max_length',
                                    max_length=gt.text_trunc_length - 1)
  text_length = np.sum(text_input['attention_mask'])

  mha_weights[cid] = mha_weights[cid][:text_length, 0, :mol_length]

  if (i+1) % 1000 == 0: print(i+1)

for i,d in enumerate(gt.generate_examples_val()):

  batch = d['input']

  cid = d['cid']#batch['molecule']['cid'][0]
  text_mask = torch.Tensor(batch['text']['attention_mask']).bool().reshape(1,-1).to(device)

  text = torch.Tensor(batch['text']['input_ids']).int().reshape(1,-1).to(device)
  graph_batch, molecule_mask = graph_batcher_val([batch['molecule']['cid']])
  graph_batch = graph_batch.to(device)
  molecule_mask = molecule_mask.to(device)
  graph_batch.edge_index = graph_batch.edge_index.reshape((2,-1))

  out = model(text, graph_batch, text_mask, molecule_mask)

  #for memory reasons
  mol_length = graph_batch.x.shape[0]
  text_input = gt.text_tokenizer(gt.descriptions[cid], truncation=True, padding = 'max_length',
                                    max_length=gt.text_trunc_length - 1)
  text_length = np.sum(text_input['attention_mask'])
  mha_weights[cid] = mha_weights[cid][:text_length, 0, :mol_length]


  if (i+1) % 1000 == 0: print(i+1)

for i,d in enumerate(gt.generate_examples_test()):

  batch = d['input']

  cid = d['cid']
  text_mask = torch.Tensor(batch['text']['attention_mask']).bool().reshape(1,-1).to(device)

  text = torch.Tensor(batch['text']['input_ids']).int().reshape(1,-1).to(device)
  graph_batch, molecule_mask = graph_batcher_test([batch['molecule']['cid']])
  graph_batch = graph_batch.to(device)
  molecule_mask = molecule_mask.to(device)
  graph_batch.edge_index = graph_batch.edge_index.reshape((2,-1))


  out = model(text, graph_batch, text_mask, molecule_mask)

  #for memory reasons
  mol_length = graph_batch.x.shape[0]
  text_input = gt.text_tokenizer(gt.descriptions[cid], truncation=True, padding = 'max_length',
                                    max_length=gt.text_trunc_length - 1)
  text_length = np.sum(text_input['attention_mask'])
  mha_weights[cid] = mha_weights[cid][:text_length, 0, :mol_length]


  if (i+1) % 1000 == 0: print(i+1)

import pickle

path = "attention_weights/"
with open(path + "mha_weights.pkl", 'wb') as fp:
  pickle.dump(mha_weights, fp)

