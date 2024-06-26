  

# 598DHL Team 21

Sherry Li (xuehail2@illinois.edu), Jo Yang (jiaoy2@illinois.edu)

## Project Draft Jupyter Notebook

Please find our final submission of the Jupyter notebook here: [DL4H_Team_21.ipynb](https://github.com/sherrylinice/598dhl/blob/main/DL4H_Team_21%20Final..ipynb). 

The 4-minitues video presentation provides an overview of the project and insights into the research findings.
[Video presentation URL](https://mediaspace.illinois.edu/media/t/1_zekf6gtq)

You may refer to the provided instructions before running it: [instruction doc for running DL4H_Team_21.ipynb](https://github.com/sherrylinice/598dhl/blob/f011712ec4e8754a0ce2969d6873fa31c8304e68/Instruction%20to%20run%20DL4H_Team21.ipynb%20(1).docx). 

Your feedback and questions are welcomed!

## Text2Mol

This is code for the paper [Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries](https://aclanthology.org/2021.emnlp-main.47/)


![Task Example](https://github.com/cnedwards/text2mol/blob/master/misc/task2.PNG?raw=true)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text2mol-cross-modal-molecule-retrieval-with/cross-modal-retrieval-on-chebi-20)](https://paperswithcode.com/sota/cross-modal-retrieval-on-chebi-20?p=text2mol-cross-modal-molecule-retrieval-with)

### Installation

Code is written in Python 3. Packages are shown in code/packages.txt. However, the following should suffice:
> pytorch
> pytorch-geometric
> transformers
> scikit-learn
> numpy

For processing .sdf files, we recommend [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html).

For ranker_threshold.py:
> matplotlib

### Files

| File      | Description |
| ----------- | ----------- |
| main.py      | Train Text2Mol.       |
| main_parallel.py   | A lightly-tested parallel version. Not used in our project.     |
| ranker.py   | Rank output embeddings.        |
| ensemble.py   | Rank ensemble of output embeddings.        |
| test_example.py   | Runs a version of the model that you can query with arbitrary inputs for testing.        |
| extract_embeddings.py   | Extract embeddings or rules from a specific checkpoint.        |
| ranker_threshold.py   | Rank output embeddings and plot cosine score vs. ranking.        |
| models.py   | The three model definitions: MLP, GCN, and Attention.        |
| losses.py   | Losses used for training.        |
| dataloaders.py   | Code for loading the data.        |
| ablation_option.py   | The ablation options included in this project draft.        |
| dataloaders_length_ablation.py   | Additional ablation study.        |
| main_sample.py   | Use a small sample for testing.        |
| attention_mrr.py | Metrics calculation for the attention model.        |
| attention_weights_recalc.py | Recalculate attention weights from an attention model checkpoint.        |
| requirements.yml   | Requirement files.        |
| notebooks   | Jupyter Notebooks/Google Collab implementations.        |


### Example commands:

To train the model:

> python code/main.py --data data --output_path test_output --model MLP --epochs 40 --batch_size 32

ranker.py can be used to rank embedding outpoints. ensemble.py ranks the ensemble of multiple embeddings.  

> python code/ranker.py test_output/embeddings --train --val --test

> python code/ensemble.py test_output/embeddings GCN_outputs/embeddings --train --val --test

To run example queries given a model checkpoint for the MLP model:

> python code/test_example.py test_output/embeddings/ data/ test_output/CHECKPOINT.pt

To get embeddings from a specific checkpoint:

> python code/extract_embeddings.py --data data --output_path embedding_output_dir --checkpoint test_output/CHECKPOINT.pt --model MLP --batch_size 32

To plot cosine score vs ranking:

> python code/ranker_threshold.py test_output/embeddings --train --val --test --output_file threshold_image.png

To run ablation study: 
> python code/main.py --data data --output_path test_output --model MLP --epochs 40 --batch_size 32 --normalization_layer_removal True

Ablation args include : normalization_layer_removal, max_pool, hidden_layer_removal, conv_layer_removal, add_dropout, change_loss, text_length_ablation.

To calculate metrics for the attention model:
> python code/attention_mrr.py --weights_dir <path to mha_weights.pkl, e.g., 'test_output/'> --embeddings_dir <path to embeddings, e.g., 'test_output/embeddings/'>

Or run the jupyter notebook: [association_rules_runnable_debugged_version.ipynb](https://github.com/sherrylinice/598dhl/blob/main/code/notebooks/association_rules_runnable_debugged_version.ipynb)

To calculate model eval metrics with FPGrowth:
> python code/fpgrowth_mrr.py --embeddings_dir <path to embeddings, e.g., 'test_output/embeddings/'>

Or run the jupyter notebook: [association_rules_runnable_debugged_version.ipynb](https://github.com/sherrylinice/598dhl/blob/main/code/notebooks/association_rules_runnable_debugged_version.ipynb)

To visualize the molecule embeddings and text embeddings using t-SNE: 
> python code/t-sne_chem.py --chem_emb_dir <path to 'chem_embeddings_train.npy'> --cid_emb_dir <path to 'cids_train.npy'>

> python code/t-sne_text.py --chem_emb_dir <path to 'text_embeddings_train.npy'> --cid_emb_dir <path to 'cids_train.npy'>

All code has been rewritten as Python files so far except association_rules.ipynb. Note: the association_rules.ipynb has been refactored to association_rules_runnable_debugged_version.ipynb. In addition, 
two separate .py files have been created to implement the attention association rules and the FPGrowth association rules, as attention_mrr.py and fpgrowth_mrr.py respectively. 


### Data: *ChEBI-20*

Data can be found in "data/". Files directly used in the dataloaders are "training.txt", "val.txt", and "test.txt". These include the CIDs (pubchem compound IDs), mol2vec embeddings, and ChEBI descriptions. SDF (structural data file) versions are also available. 

Thanks to [PubChem](https://pubchem.ncbi.nlm.nih.gov/) and [ChEBI](https://www.ebi.ac.uk/chebi/) for freely providing access to their databases. 


### Citation
If you found our work useful, please cite:
```bibtex
@inproceedings{edwards2021text2mol,
  title={Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries},
  author={Edwards, Carl and Zhai, ChengXiang and Ji, Heng},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={595--607},
  year={2021},
  url = {https://aclanthology.org/2021.emnlp-main.47/}
}
```


![Poster](https://github.com/cnedwards/text2mol/blob/master/misc/Text2Mol_EMNLP_poster.png?raw=true)
