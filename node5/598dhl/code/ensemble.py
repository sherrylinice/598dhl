
#Note: There is a very slight variance in the ensemble results which arises because the molecule encoder cannot distinguish things like isotopes.

#python ensemble.py ../softmax_CLIP/embeddings/GCN1/ ../softmax_CLIP/embeddings/GCN2/ ../softmax_CLIP/embeddings/GCN3/ --test

import os
import os.path as osp
import shutil
import math

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import argparse

parser = argparse.ArgumentParser(description='Ensemble of Text2mol embeddings.')
parser.add_argument('dirs', metavar='directories', type=str, nargs='+',
                    help='all directories where embeddings are located')
parser.add_argument('--train', action='store_const', const=True, help="calculate training split ranks")
parser.add_argument('--val', action='store_const', const=True, help="calculate validation split ranks")
parser.add_argument('--test', action='store_const', const=True, help="calculate test split ranks")

args = parser.parse_args()
dirs = list(args.dirs)

'''
inputs = "../softmax_CLIP/embeddings"
dir1 = osp.join(inputs, "GCN1/")
dir2 = osp.join(inputs, "GCN2/")
dir3 = osp.join(inputs, "GCN3/")
dir4 = osp.join(inputs, "MLP1/")
dir5 = osp.join(inputs, "MLP2/")
dir6 = osp.join(inputs, "MLP3/")
dirs = [dir1, dir2, dir3, dir4, dir5, dir6]
'''

NUM_MODELS = len(dirs)


text_embeddings_train = []
text_embeddings_val = []
text_embeddings_test = []
chem_embeddings_train = []
chem_embeddings_val = []
chem_embeddings_test = []

cids_train = []
cids_val = []
cids_test = []

for i, dir in enumerate(dirs):

    cids_train.append(np.load(osp.join(dir, "cids_train.npy"), allow_pickle=True))
    cids_val.append(np.load(osp.join(dir, "cids_val.npy"), allow_pickle=True))
    cids_test.append(np.load(osp.join(dir, "cids_test.npy"), allow_pickle=True))

    text_embeddings_train.append(np.load(osp.join(dir, "text_embeddings_train.npy")))
    text_embeddings_val.append(np.load(osp.join(dir, "text_embeddings_val.npy")))
    text_embeddings_test.append(np.load(osp.join(dir, "text_embeddings_test.npy")))

    chem_embeddings_train.append(np.load(osp.join(dir, "chem_embeddings_train.npy")))
    chem_embeddings_val.append(np.load(osp.join(dir, "chem_embeddings_val.npy")))
    chem_embeddings_test.append(np.load(osp.join(dir, "chem_embeddings_test.npy")))

    print('Loaded embedding from model', i+1)

print('Loaded embeddings')

#Reorder (this is very important):

for i in range(1, NUM_MODELS):
    tmp = cids_train[i].tolist()
    indexes = [tmp.index(i) for i in cids_train[0]]
    tmp = cids_val[i].tolist()
    indexes_val = [tmp.index(i) for i in cids_val[0]]
    tmp = cids_test[i].tolist()
    indexes_test = [tmp.index(i) for i in cids_test[0]]

    cids_train[i] = cids_train[i][indexes]
    cids_val[i] = cids_val[i][indexes_val]
    cids_test[i] = cids_test[i][indexes_test]

    text_embeddings_train[i] = text_embeddings_train[i][indexes]
    text_embeddings_val[i] = text_embeddings_val[i][indexes_val]
    text_embeddings_test[i] = text_embeddings_test[i][indexes_test]
    
    chem_embeddings_train[i] = chem_embeddings_train[i][indexes]
    chem_embeddings_val[i] = chem_embeddings_val[i][indexes_val]
    chem_embeddings_test[i] = chem_embeddings_test[i][indexes_test]

    print('Embeddings {} reordered'.format(i+1))

print('Sorted embeddings')

#combine all splits:
all_text_embbedings = []
all_mol_embeddings = []
for i in range(NUM_MODELS):
    all_text_embbedings.append(np.concatenate((text_embeddings_train[i], text_embeddings_val[i], text_embeddings_test[i]), axis = 0))
    all_mol_embeddings.append(np.concatenate((chem_embeddings_train[i], chem_embeddings_val[i], chem_embeddings_test[i]), axis = 0))
    
all_cids = np.concatenate((cids_train[0], cids_val[0], cids_test[0]), axis = 0)


n_train = len(cids_train[0])
n_val = len(cids_val[0])
n_test = len(cids_test[0])
n = n_train + n_val + n_test

offset_val = n_train
offset_test = n_train + n_val


#I wrote a multithreaded version of the cosine similarity for something else. I can upload it if needed.

#Create efficient cosine calculator
def memory_efficient_similarity_matrix_custom(func, embedding1, embedding2, chunk_size = 1000):
    rows = embedding1.shape[0]
    
    num_chunks = int(np.ceil(rows / chunk_size))
    
    for i in range(num_chunks):
        end_chunk = (i+1)*(chunk_size) if (i+1)*(chunk_size) < rows else rows #account for smaller chunk at end...
        yield func(embedding1[i*chunk_size:end_chunk,:], embedding2)

text_chem_cos = []
text_chem_cos_val = []
text_chem_cos_test = []
for i in range(NUM_MODELS):
    text_chem_cos.append(memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_train[i], all_mol_embeddings[i]))
    text_chem_cos_val.append(memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_val[i], all_mol_embeddings[i]))
    text_chem_cos_test.append(memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_test[i], all_mol_embeddings[i]))

#Calculate Ranks:
if args.train:
    tr_avg_ranks = np.zeros((n_train, n))
if args.val:
    val_avg_ranks = np.zeros((n_val, n))
if args.test:
    test_avg_ranks = np.zeros((n_test, n))

ranks_train = []
ranks_val = []
ranks_test = []

def get_ranks(text_chem_cos, ranks_avg, offset, split= ""):
    ranks_tmp = []
    j = 0 #keep track of all loops
    for l, emb in enumerate(text_chem_cos):
        for k in range(emb.shape[0]):
            cid_locs = np.argsort(emb[k,:])[::-1]
            ranks = np.argsort(cid_locs) 
            
            # Additional experiment: change rank sum to maximum rank. add the next line.  
            ranks_avg[j,:] = np.maximum(ranks_avg[j,:], ranks)
            #ranks_avg[j,:] = ranks_avg[j,:] + ranks 
            
            rank = ranks[j+offset] + 1
            ranks_tmp.append(rank)
            

            j += 1
            if j % 1000 == 0: print(j, split+" processed")

    return np.array(ranks_tmp)

def print_ranks(ranks, model_num, split):

    print(split+" Model {}:".format(model_num))
    print("Mean rank:", np.mean(ranks))
    print("Hits at 1:", np.mean(ranks <= 1))
    print("Hits at 10:", np.mean(ranks <= 10))
    print("Hits at 100:", np.mean(ranks <= 100))
    print("Hits at 500:", np.mean(ranks <= 500))
    print("Hits at 1000:", np.mean(ranks <= 1000))

    print("MRR:", np.mean(1/ranks))
    print()


for i in range(NUM_MODELS):
    if args.train:
        ranks_tmp = get_ranks(text_chem_cos[i], tr_avg_ranks, offset=0, split="train")
        print_ranks(ranks_tmp, i+1, split="Training")
        ranks_train.append(ranks_tmp)

    if args.val:
        ranks_tmp = get_ranks(text_chem_cos_val[i], val_avg_ranks, offset=offset_val, split="val")
        print_ranks(ranks_tmp, i+1, split="Validation")
        ranks_val.append(ranks_tmp)
    
    if args.test:
        ranks_tmp = get_ranks(text_chem_cos_test[i], test_avg_ranks, offset=offset_test, split="test")
        print_ranks(ranks_tmp, i+1, split="Test")
        ranks_test.append(ranks_tmp)


# Original ensemble strategy:
#Process ensemble:

if args.train:
    sorted = np.argsort(tr_avg_ranks)
    new_tr_ranks = np.diag(np.argsort(sorted)) + 1
    print_ranks(new_tr_ranks, "e", split="Training Ensemble")

if args.val:
    sorted = np.argsort(val_avg_ranks)
    val_final_ranks = np.argsort(sorted) + 1
    new_val_ranks = np.diag(val_final_ranks[:,offset_val:offset_test])

    print_ranks(new_val_ranks, "e", split="Validation Ensemble")

if args.test:
    sorted = np.argsort(test_avg_ranks)
    test_final_ranks = np.argsort(sorted) + 1
    new_test_ranks = np.diag(test_final_ranks[:,offset_test:])

    print_ranks(new_test_ranks, "e", split="Test Ensemble")

# Additional experiment 1: rank average ensemble
# Calculate ensemble using rank average
"""
def calculate_ensemble_rank_average(ranks_list):
    ensemble_ranks = np.mean(ranks_list, axis=0)
    return ensemble_ranks

# Process ensemble using rank average
if args.train:
    ensemble_train_ranks = calculate_ensemble_rank_average(ranks_train)
    print_ranks(ensemble_train_ranks, 'Rank Average Ensemble', split="Training")

if args.val:
    ensemble_val_ranks = calculate_ensemble_rank_average(ranks_val)
    print_ranks(ensemble_val_ranks, 'Rank Average Ensemble', split="Validation")

if args.test:
    ensemble_test_ranks = calculate_ensemble_rank_average(ranks_test)
    print_ranks(ensemble_test_ranks, 'Rank Average Ensemble', split="Test")

# Additional experiment 2: Weighted rank average ensemble
# Calculate ensemble using weighted rank average

def calculate_ensemble_weighted_rank_average(ranks_list, weights):
    weighted_ranks = [ranks * weight for ranks, weight in zip(ranks_list, weights)]
    ensemble_ranks = np.sum(weighted_ranks, axis=0)
    return ensemble_ranks

# Process ensemble using weighted rank average
if args.train:
    train_mrr_list = [np.mean(1 / ranks) for ranks in ranks_train]
    train_weights = [mrr / sum(train_mrr_list) for mrr in train_mrr_list]
    ensemble_train_ranks = calculate_ensemble_weighted_rank_average(ranks_train, train_weights)
    print_ranks(ensemble_train_ranks, 'Weighted Rank Average Ensemble', split="Training")

if args.val:
    val_mrr_list = [np.mean(1 / ranks) for ranks in ranks_val]
    val_weights = [mrr / sum(val_mrr_list) for mrr in val_mrr_list]
    ensemble_val_ranks = calculate_ensemble_weighted_rank_average(ranks_val, val_weights)
    print_ranks(ensemble_val_ranks, 'Weighted Rank Average Ensemble', split="Validation")

if args.test:
    test_mrr_list = [np.mean(1 / ranks) for ranks in ranks_test]
    test_weights = [mrr / sum(test_mrr_list) for mrr in test_mrr_list]
    ensemble_test_ranks = calculate_ensemble_weighted_rank_average(ranks_test, test_weights)
    print_ranks(ensemble_test_ranks, 'Weighted Rank Average Ensemble', split="Test")

# Additional experiment 3: weighted rank average and relative rankings
# Calculate ensemble using weighted rank average and relative rankings

def calculate_ensemble_weighted_rank_average_relative(ranks_list, weights):
    # Convert ranks to relative rankings
    relative_ranks_list = []
    for ranks in ranks_list:
        relative_ranks = np.argsort(ranks) + 1
        relative_ranks_list.append(relative_ranks)
    
    # Calculate weighted relative ranks
    weighted_relative_ranks = [relative_ranks * weight for relative_ranks, weight in zip(relative_ranks_list, weights)]
    ensemble_relative_ranks = np.sum(weighted_relative_ranks, axis=0)
    
    # Convert ensemble relative ranks back to actual ranks
    ensemble_ranks = np.argsort(ensemble_relative_ranks) + 1
    
    return ensemble_ranks

# Process ensemble using weighted rank average and relative rankings
if args.train:
    train_mrr_list = [np.mean(1 / ranks) for ranks in ranks_train]
    train_weights = [mrr / sum(train_mrr_list) for mrr in train_mrr_list]
    ensemble_train_ranks = calculate_ensemble_weighted_rank_average_relative(ranks_train, train_weights)
    print_ranks(ensemble_train_ranks, 'Weighted Rank Average Ensemble (Relative)', split="Training")

if args.val:
    val_mrr_list = [np.mean(1 / ranks) for ranks in ranks_val]
    val_weights = [mrr / sum(val_mrr_list) for mrr in val_mrr_list]
    ensemble_val_ranks = calculate_ensemble_weighted_rank_average_relative(ranks_val, val_weights)
    print_ranks(ensemble_val_ranks, 'Weighted Rank Average Ensemble (Relative)', split="Validation")

if args.test:
    test_mrr_list = [np.mean(1 / ranks) for ranks in ranks_test]
    test_weights = [mrr / sum(test_mrr_list) for mrr in test_mrr_list]
    ensemble_test_ranks = calculate_ensemble_weighted_rank_average_relative(ranks_test, test_weights)
    print_ranks(ensemble_test_ranks, 'Weighted Rank Average Ensemble (Relative)', split="Test")

# Calculate weighted average ranks
if args.train:
    train_mrr_list = [np.mean(1 / ranks) for ranks in ranks_train]
    train_weights = [mrr / sum(train_mrr_list) for mrr in train_mrr_list]
    tr_avg_ranks = np.zeros((n_train, n))
    for i in range(NUM_MODELS):
        tr_avg_ranks += np.argsort(ranks_train[i]) * train_weights[i]
        
if args.val:
    val_mrr_list = [np.mean(1 / ranks) for ranks in ranks_val]
    val_weights = [mrr / sum(val_mrr_list) for mrr in val_mrr_list]
    val_avg_ranks = np.zeros((n_val, n))
    for i in range(NUM_MODELS):
        val_avg_ranks += np.argsort(ranks_val[i]) * val_weights[i]
        
if args.test:
    test_mrr_list = [np.mean(1 / ranks) for ranks in ranks_test]
    test_weights = [mrr / sum(test_mrr_list) for mrr in test_mrr_list]
    test_avg_ranks = np.zeros((n_test, n))
    for i in range(NUM_MODELS):
        test_avg_ranks += np.argsort(ranks_test[i]) * test_weights[i]

# Process ensemble using weighted average ranks and relative rankings
if args.train:
    sorted_train = np.argsort(tr_avg_ranks)
    ensemble_train_ranks = np.diag(np.argsort(sorted_train)) + 1
    print_ranks(ensemble_train_ranks, "Weighted Average Ensemble (Relative)", split="Training")

if args.val:
    sorted_val = np.argsort(val_avg_ranks)
    ensemble_val_ranks = np.diag(np.argsort(sorted_val)[:,offset_val:offset_test]) + 1
    print_ranks(ensemble_val_ranks, "Weighted Average Ensemble (Relative)", split="Validation")

if args.test:
    sorted_test = np.argsort(test_avg_ranks)
    ensemble_test_ranks = np.diag(np.argsort(sorted_test)[:,offset_test:]) + 1
    print_ranks(ensemble_test_ranks, "Weighted Average Ensemble (Relative)", split="Test")
"""
