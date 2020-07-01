# Introduction
This is the code base for paper "Countering Language Drift with Iterated Learning". Please cite the following if you found this codebase to be useful
```
@article{lu2020countering,
  title={Countering language drift with seeded iterated learning},
  author={Lu, Yuchen and Singhal, Soumye and Strub, Florian and Pietquin, Olivier and Courville, Aaron},
  journal={arXiv preprint arXiv:2003.12694},
  year={2020}
}
```
This codebase is tested on `python3.7` and `torch==1.3.1`.

# Install
Clone this repo, and at project root do `pip install -e .` as well as other dependencies.

# Prepare Datasets
```
python preprocess/prepare_text.py -data_dir DATA_DIR
```
Put the images of multi30k under the directory `DATA_DIR/multi30k/imgs`,
then do
```
python preprocess/extract_flickr30_imgfeats.py -datadir DATA_DIR
```
This will produce `train_feat.pth` and `val_feat.pth` under `DATA_DIR/flickr30k`.

# Pretrain & Finetune
```
python run_pretrain.py --config JSON_PATH --data_dir DATA_DIR --exp_dir EXP_ROOT_DIR
```
```
python run_finetune.py --config JSON_PATH --data_dir DATA_DIR --exp_dir EXP_ROOT_DIR
```

# JSON Configs
Sample json config can be found under the folder `jsons`.
1. `iwslt_en_de.json`, `iwslt_fr_en.json`: Config for pretraining translation agents.
2. `hyperparames_caption.json`: Config for language model used for scoring generated English.
3. `hyperparames_ranker.json`: Config for pretraining image retriever.
5. `gumbel/`: Configs for Gumbel finetuning (vanilla Gumbel, SIL, S2P)
