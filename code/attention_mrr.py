
import pickle
import csv

import numpy as np
from transformers import BertTokenizerFast, BertModel

import operator
from collections import defaultdict


# special generator for random sampling

#Need a special generator for random sampling:
import argparse

parser = argparse.ArgumentParser(description='Run attention MRR calculation')
parser.add_argument('--weights_dir', metavar='weights_dir', type=str, 
                    help='directory where weights is located')
parser.add_argument('--embeddings_dir', metavar='embeddings_dir', type=str, 
                    help='directory where embeddings is located')
args = parser.parse_args()  
weights_path = args.weights_dir
embeddings_path = args.embeddings_dir

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

    self.validation_cids = []
    #get validation set cids...
    with open(self.path_val) as f:
      reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
      for n, line in enumerate(reader):
        self.descriptions[line['cid']] = line['desc']
        self.mols[line['cid']] = line['mol2vec']
        self.validation_cids.append(line['cid'])

    self.test_cids = []
    with open(self.path_test) as f:
      reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
      for n, line in enumerate(reader):
        self.descriptions[line['cid']] = line['desc']
        self.mols[line['cid']] = line['mol2vec']
        self.test_cids.append(line['cid'])

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

graph_data_path = "data/mol_graphs.zip"
dir_emb = embeddings_path
mounted_path_token_embs = "data/token_embedding_dict.npy"
mounted_path_train = "data/training.txt"
mounted_path_val = "data/val.txt"
mounted_path_test = "data/test.txt"
mounted_path_molecules = "data/ChEBI_definitions_substructure_corpus.cp"
gt = GenerateData(mounted_path_train, mounted_path_val, mounted_path_test, mounted_path_molecules, mounted_path_token_embs)

cids_train1 = np.load(dir_emb + "cids_train.npy", allow_pickle=True)
cids_val1 = np.load(dir_emb + "cids_val.npy", allow_pickle=True)
cids_test1 = np.load(dir_emb + "cids_test.npy", allow_pickle=True)
chem_embeddings_train1 = np.load(dir_emb + "chem_embeddings_train.npy")
chem_embeddings_val1 = np.load(dir_emb + "chem_embeddings_val.npy")
chem_embeddings_test1 = np.load(dir_emb + "chem_embeddings_test.npy")
text_embeddings_train1 = np.load(dir_emb + "text_embeddings_train.npy")
text_embeddings_val1 = np.load(dir_emb + "text_embeddings_val.npy")
text_embeddings_test1 = np.load(dir_emb + "text_embeddings_test.npy")


all_chem_embbedings1 = np.concatenate((chem_embeddings_train1, chem_embeddings_val1, chem_embeddings_test1), axis = 0)

cids_all = np.concatenate((cids_train1, cids_val1, cids_test1), axis = 0)

from sklearn.metrics.pairwise import cosine_similarity

def memory_efficient_similarity_matrix_custom(func, embedding1, embedding2, chunk_size = 1000):
    rows = embedding1.shape[0]

    num_chunks = int(np.ceil(rows / chunk_size))

    for i in range(num_chunks):
        end_chunk = (i+1)*(chunk_size) if (i+1)*(chunk_size) < rows else rows #account for smaller chunk at end...
        yield func(embedding1[i*chunk_size:end_chunk,:], embedding2)

#Calculate mean rank, hits at ten

def dot_product(a, b):
  return np.dot(a, b.T)

sigmoid = lambda x: 1 / (1 + np.exp(-x))

compose = lambda a,b: sigmoid(dot_product(a,b))

text_chem_cos1 = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_train1, all_chem_embbedings1)
text_chem_cos_val1 = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_val1, all_chem_embbedings1)
text_chem_cos_test1 = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_test1, all_chem_embbedings1)

n_train = len(cids_train1)
n_val = len(cids_val1)
n_test = len(cids_test1)
n = n_train + n_val + n_test

offset_val = n_train
offset_test = n_train + n_val

num_top = 10
top_cids1 = {}
top_cids_val1 = {}
top_cids_test1 = {}
scores_val1 = {}
scores_test1 = {}

ranks1 = []
j = 0 #keep track of all loops
for i, emb in enumerate(text_chem_cos1):
    for k in range(emb.shape[0]):
        cid_locs = np.argsort(emb[k,:])[::-1]
        ranks = np.argsort(cid_locs) #rank is actually double argsort...

        top_cids1[cids_train1[j]] = [cids_all[loc] for loc in cid_locs[:num_top]]

        rank = ranks[j] + 1
        ranks1.append(rank)


        j += 1
        if (j) % 1000 == 0: print((j), "train processed.")

ranks1 = np.array(ranks1)

print()
print("MLP Training Mean rank:", np.mean(ranks1))
print("MLP Hits at 1:", np.mean(ranks1 <= 1))
print("MLP Hits at 10:", np.mean(ranks1 <= 10))
print("MLP Hits at 100:", np.mean(ranks1 <= 100))
print("MLP Hits at 500:", np.mean(ranks1 <= 500))
print("MLP Hits at 1000:", np.mean(ranks1 <= 1000))

print("MLP Trainng MRR:", np.mean(1/np.array(ranks1)))

ranks_val1 = []
j = 0 #keep track of all loops
for i, emb in enumerate(text_chem_cos_val1):
    for k in range(emb.shape[0]):
        cid_locs = np.argsort(emb[k,:])[::-1]
        ranks = np.argsort(cid_locs) #rank is actually double argsort...

        scores = np.sort(emb[k,:])[::-1]

        top_cids_val1[cids_val1[j]] = [cids_all[loc] for loc in cid_locs[:num_top]]
        scores_val1[cids_val1[j]] = scores[:num_top]

        rank = ranks[j+offset_val] + 1
        ranks_val1.append(rank)

        j += 1
        if (j) % 1000 == 0: print((j), "val processed.")


ranks_val1 = np.array(ranks_val1)

print()
print("MLP Val Mean rank:", np.mean(ranks_val1))
print("MLP Hits at 1:", np.mean(ranks_val1 <= 1))
print("MLP Hits at 10:", np.mean(ranks_val1 <= 10))
print("MLP Hits at 100:", np.mean(ranks_val1 <= 100))
print("MLP Hits at 500:", np.mean(ranks_val1 <= 500))
print("MLP Hits at 1000:", np.mean(ranks_val1 <= 1000))

print("MLP Validation MRR:", np.mean(1/ranks_val1))


ranks_test1 = []
j = 0 #keep track of all loops
for i, emb in enumerate(text_chem_cos_test1):
    for k in range(emb.shape[0]):
        cid_locs = np.argsort(emb[k,:])[::-1]
        ranks = np.argsort(cid_locs) #rank is actually double argsort...

        scores = np.sort(emb[k,:])[::-1]

        top_cids_test1[cids_test1[j]] = [cids_all[loc] for loc in cid_locs[:num_top]]
        scores_test1[cids_test1[j]] = scores[:num_top]

        rank = ranks[j+offset_test] + 1
        ranks_test1.append(rank)

        j += 1
        if (j) % 1000 == 0: print((j), "test processed.")


ranks_test1 = np.array(ranks_test1)

print()
print("MLP Test Mean rank:", np.mean(ranks_test1))
print("MLP Hits at 1:", np.mean(ranks_test1 <= 1))
print("MLP Hits at 10:", np.mean(ranks_test1 <= 10))
print("MLP Hits at 100:", np.mean(ranks_test1 <= 100))
print("MLP Hits at 500:", np.mean(ranks_test1 <= 500))
print("MLP Hits at 1000:", np.mean(ranks_test1 <= 1000))

print("MLP Test MRR:", np.mean(1/ranks_test1))


# get attention rules
with open(f'{weights_path}/mha_weights.pkl', 'rb') as f:
    mha_weights = pickle.load(f)

# for i, cid in enumerate(mha_weights):
#     attn_weights = mha_weights[cid]
#     # if attn_weights.shape[0]!=1:
#     #     print(i)
#     if i<=5:
#         print(attn_weights.shape)

all_mol_tokens = set()
all_text_tokens = set()

import zipfile
archive = zipfile.ZipFile(graph_data_path, 'r')

for i, cid in enumerate(mha_weights):
  attn_weights = mha_weights[cid]
  text_input = gt.text_tokenizer(gt.descriptions[cid], truncation=True, padding = 'max_length',
                                    max_length=gt.text_trunc_length - 1)
  text_length = np.sum(text_input['attention_mask'])
  text_tokens = gt.text_tokenizer.convert_ids_to_tokens(text_input['input_ids'][:text_length])

  gfile = archive.open(cid + '.graph').read().decode('ascii')
  mol_tokens = {}
  idx = False
  for line in gfile.split('\n'):
    line = line.strip()
    if line == 'idx to identifier:':
      idx = True
      continue
    if idx and len(line) != 0:
      id, idf = line.split(" ")
      mol_tokens[id] = idf

  mol_tokens = list(mol_tokens.values())

  all_mol_tokens.update(mol_tokens)
  all_text_tokens.update(text_tokens)

mol_token_ids = {}
text_token_ids = {}

mol_token_ids_rev = {}
text_token_ids_rev = {}
for i, k in enumerate(all_mol_tokens):
  mol_token_ids[k] = i
  mol_token_ids_rev[i] = k
for i, k in enumerate(all_text_tokens):
  text_token_ids[k] = i
  text_token_ids_rev[i] = k

support = np.zeros((len(all_text_tokens), len(all_mol_tokens)))
conf = np.zeros((len(all_text_tokens), len(all_mol_tokens)))

for i, cid in enumerate(mha_weights):
  # print('cid:', cid)
  if cid in gt.validation_cids or cid in gt.test_cids: continue
  attn_weights = mha_weights[cid]
  text_input = gt.text_tokenizer(gt.descriptions[cid], truncation=True, padding = 'max_length',
                                    max_length=gt.text_trunc_length - 1)
  text_length = np.sum(text_input['attention_mask'])
  text_tokens = gt.text_tokenizer.convert_ids_to_tokens(text_input['input_ids'][:text_length])

  gfile = archive.open(cid + '.graph').read().decode('ascii')
  mol_tokens = {}
  idx = False
  for line in gfile.split('\n'):
    line = line.strip()
    if line == 'idx to identifier:':
      idx = True
      continue
    if idx and len(line) != 0:
      id, idf = line.split(" ")
      mol_tokens[id] = idf
  mol_tokens = list(mol_tokens.values())

  if len(mol_tokens) > gt.mol_trunc_length: mol_tokens = mol_tokens[:gt.mol_trunc_length]

  # print(len(text_tokens))
  # print(len(mol_tokens))
  for j, text in enumerate(text_tokens):
    for k, molt in enumerate(mol_tokens):
      # print(j, k, attn_weights[j,k])
      # print(support[text_token_ids[text], mol_token_ids[molt]])
      # if j <= attn_weights.shape[0] - 1:
      support[text_token_ids[text], mol_token_ids[molt]] += attn_weights[j,k] #* mol_length # mol_length to normalize


  if (i+1) % 1000 == 0: print(i+1)

print("Support calculation finished.")

for j, text in enumerate(all_text_tokens):
  if np.sum(support[text_token_ids[text], :]) == 0:
    conf[text_token_ids[text], :] = 0.0
  else:
    conf[text_token_ids[text], :] = support[text_token_ids[text], :] / np.sum(support[text_token_ids[text], :])

  if (j+1) % 1000 == 0: print(j+1)

print("Confidence calculation finished.")

from itertools import combinations, chain


def all_subsets(ss):#skip empty set
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1)))


def generate_rules(text_tokens, mol_tokens):
  candidates = set()

  text_subs = [frozenset([text_token_ids[j] for j in i]) for i in combinations(text_tokens, 1)]
  mol_subs = [frozenset([mol_token_ids[j] for j in i]) for i in combinations(mol_tokens, 1)]

  rules = []

  for t in text_subs:
    for m in mol_subs:
      rules.append((t, m))

  return rules


def ar_score(text_cid, mol_cid, top_num=10):

  text_input = gt.text_tokenizer(gt.descriptions[text_cid], truncation=True, padding = 'max_length',
                                    max_length=gt.text_trunc_length - 1)
  text_length = np.sum(text_input['attention_mask'])
  text_tokens = gt.text_tokenizer.convert_ids_to_tokens(text_input['input_ids'][:text_length])

  gfile = archive.open(mol_cid + '.graph').read().decode('ascii')
  mol_tokens = {}
  idx = False
  for line in gfile.split('\n'):
    line = line.strip()
    if line == 'idx to identifier:':
      idx = True
      continue
    if idx and len(line) != 0:
      id, idf = line.split(" ")
      mol_tokens[id] = idf
  mol_tokens = list(mol_tokens.values())

  rules = generate_rules(text_tokens, mol_tokens)

  tmp = np.array([conf[list(r[0])[0], list(r[1])[0]] for r in rules])

  mx = np.min((top_num, len(tmp)))
  top_confs = -np.partition(-tmp, mx-1)[:mx]


  return np.mean(top_confs)

# calculate alpha using validataion dataset, alpha = 0
ar_scores = np.zeros((len(top_cids_val1), num_top))

# alpha 0 - 101
x = np.linspace(0.0,1,101)
MRRs = []
hits1 = []
hits10 = []


for n in x:
  
  alpha = n
  print("alpha:", alpha)
  hits_at_one = 0
  hits_at_ten = 0
  hits_at_100 = 0

  tmp_ranks = []
  for i, cid in enumerate(top_cids_val1):

    score = np.zeros((num_top))
    for j, cid2 in enumerate(top_cids_val1[cid]):
      tmp = ar_score(cid, cid2)
      ar_scores[i,j] = tmp
      score[j] = alpha * scores_val1[cid][j] + (1 - alpha) * ar_scores[i,j]

    try:
      old_loc = top_cids_val1[cid].index(cid)

      sorted = np.argsort(-score, kind='stable')

      new_rank = np.where(sorted == old_loc)[0][0] + 1

    except ValueError:
      new_rank = ranks_val1[i]

    tmp_ranks.append(new_rank)

    if new_rank <= 1:
        hits_at_one += 1
    if new_rank <= 10:
        hits_at_ten += 1
    if new_rank <= 100:
        hits_at_100 += 1
  print("MRR", np.mean(1/np.array(tmp_ranks)))

  MRRs.append(np.mean(1/np.array(tmp_ranks)))
  hits1.append(hits_at_one/cids_val1.size)
  hits10.append(hits_at_ten/cids_val1.size)

print("Val Mean rank:", np.mean(tmp_ranks))
print("Hits at 1:", hits_at_one/cids_val1.size)
print("Hits at 10:", hits_at_ten/cids_val1.size)
print("Hits at 100:", hits_at_100/cids_val1.size)

print("Validation MRR:", np.mean(1/np.array(tmp_ranks)))

import operator
from collections import defaultdict

alpha = x[np.argmax(MRRs)]

ar_scores_test = np.zeros((len(top_cids_test1), num_top))

new_ranks_test = []
for i, cid in enumerate(top_cids_test1):

  text_input = gt.text_tokenizer(gt.descriptions[cid], truncation=True, padding = 'max_length',
                                    max_length=gt.text_trunc_length - 1)
  text_length = np.sum(text_input['attention_mask'])
  text_tokens = gt.text_tokenizer.convert_ids_to_tokens(text_input['input_ids'][:text_length])

  score = np.zeros((num_top))
  for j, cid2 in enumerate(top_cids_test1[cid]):
    gfile = archive.open(cid + '.graph').read().decode('ascii')
    mol_tokens = {}
    idx = False
    for line in gfile.split('\n'):
      line = line.strip()
      if line == 'idx to identifier:':
        idx = True
        continue
      if idx and len(line) != 0:
        id, idf = line.split(" ")
        mol_tokens[id] = idf
    mol_tokens = list(mol_tokens.values())

    tmp = ar_score(cid, cid2)
    ar_scores_test[i,j] = tmp
    score[j] = alpha * scores_test1[cid][j] + (1 - alpha) * tmp

  try:
    old_loc = top_cids_test1[cid].index(cid)

    sorted = np.argsort(-score, kind='stable')

    new_rank = np.where(sorted == old_loc)[0][0] + 1

  except ValueError:
    new_rank = ranks_test1[i]

  new_ranks_test.append(new_rank)


  if (i+1) % 200 == 0: print(i+1)

new_ranks_test = np.array(new_ranks_test)

print()
print("Test Mean rank:", np.mean(new_ranks_test))
print("Hits at 1:", np.mean(new_ranks_test <= 1))
print("Hits at 10:", np.mean(new_ranks_test <= 10))
print("Hits at 100:", np.mean(new_ranks_test <= 100))

print("Test MRR:", np.mean(1/np.array(new_ranks_test)))




