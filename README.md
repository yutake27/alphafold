# AlphaFold

This repository is based on [localcolabfold](https://github.com/YoshitakaMo/localcolabfold), and allows you to run [ColabFold](https://github.com/sokrypton/ColabFold)/AlphaFold2_advanced.ipynb locally.

It has some features that ColabFold and AlphaFold does not have.

It can output the all structures during the recycling process.
For example, if 3 is specified for the recycle number, a total of three structures (1,2,3 recycles) will be output.

It is also possible to specify all models (model_[1-5] and model_[1-5]_ptm) or multiple ensembles (1 and 8).

Finally, aggregate the reliability scores (pLDDT and pTM) for all predicted structures and write them to a csv.

## Setup

The following command will download the model parameters and create the miniconda environment for running AlphaFold.

```bash
bash install.sh
```

The miniconda python environment will be created in `colabfold-conda/bin/python3.7`.
This will not overwrite the existing conda environment.


## Usage

```
$ colabfold-conda/bin/python3.7 --help

usage: runner_af2advanced.py [-h] -i INPUT [-o OUTPUT_DIR] [-ho HOMOOLIGOMER]
                             [-m {mmseqs2,single_sequence,precomputed}]
                             [--precomputed PRECOMPUTED]
                             [-p {unpaired,unpaired+paired,paired}]
                             [-pc PAIR_COV] [-pq PAIR_QID]
                             [-b {pLDDT,pTMscore}] [--noranking] [-t]
                             [-mm MAX_MSA]
                             [--num_CASP14_models NUM_CASP14_MODELS]
                             [--num_pTM_models NUM_PTM_MODELS]
                             [--model {CASP14,pTM,both}]
                             [-e [{1,8} [{1,8} ...]]] [-r MAX_RECYCLES]
                             [--output_all_cycle] [--tol TOL] [--is_training]
                             [--num_samples NUM_SAMPLES]
                             [--num_relax {None,Top1,Top5,All}]
                             [--save_images] [--show_images]

Runner script that can take command-line arguments

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to a FASTA file. Required.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to a directory that will store the results. The
                        default name is 'prediction_<hash>'.
  -ho HOMOOLIGOMER, --homooligomer HOMOOLIGOMER
                        homooligomer: Define number of copies in a homo-
                        oligomeric assembly. For example, sequence:ABC:DEF,
                        homooligomer: 2:1, the first protein ABC will be
                        modeled as a omodimer (2 copies) and second DEF a
                        monomer (1 copy). Default is 1.
  -m {mmseqs2,single_sequence,precomputed}, --msa_method {mmseqs2,single_sequence,precomputed}
                        Options to generate MSA.mmseqs2 - FAST method from
                        ColabFold (default) single_sequence - use single
                        sequence input.precomputed - specify 'msa.pickle' file
                        generated previously if you have.Default is 'mmseqs2'.
  --precomputed PRECOMPUTED
                        Specify the file path of a precomputed pickled msa
                        from previous run.
  -p {unpaired,unpaired+paired,paired}, --pair_mode {unpaired,unpaired+paired,paired}
                        Experimental option for protein complexes. Pairing
                        currently only supported for proteins in same operon
                        (prokaryotic genomes). unpaired - generate seperate
                        MSA for each protein. (default) unpaired+paired -
                        attempt to pair sequences from the same operon within
                        the genome. paired - only use sequences that were
                        sucessfully paired. Default is 'unpaired'.
  -pc PAIR_COV, --pair_cov PAIR_COV
                        Options to prefilter each MSA before pairing. It might
                        help if there are any paralogs in the complex.
                        prefilter each MSA to minimum coverage with query (%)
                        before pairing. Default is 50.
  -pq PAIR_QID, --pair_qid PAIR_QID
                        Options to prefilter each MSA before pairing. It might
                        help if there are any paralogs in the complex.
                        prefilter each MSA to minimum sequence identity with
                        query (%) before pairing. Default is 20.
  -b {pLDDT,pTMscore}, --rank_by {pLDDT,pTMscore}
                        specify metric to use for ranking models (For protein-
                        protein complexes, we recommend pTMscore). Default is
                        'pLDDT'.
  --noranking           Ranking output structures. If True, the output file
                        name contains a ranking.
  -t, --use_turbo       introduces a few modifications (compile once, swap
                        params, adjust max_msa) to speedup and reduce memory
                        requirements. Disable for default behavior.
  -mm MAX_MSA, --max_msa MAX_MSA
                        max_msa defines: max_msa_clusters:max_extra_msa number
                        of sequences to use. This option ignored if use_turbo
                        is disabled. Default is '512:1024'.
  --num_CASP14_models NUM_CASP14_MODELS
                        specify how many CASP14 model (normal model) params to
                        try. (Default is 5)
  --num_pTM_models NUM_PTM_MODELS
                        specify how many pTM model params to try. (Default is
                        5)
  --model {CASP14,pTM,both}
                        Model to use. CASP14 is a normal model. Both uses both
                        models. Number of models to be used is read from
                        num_CASP14_models and num_pTM_models respectively.
                        (Default is CASP14)
  -e [{1,8} [{1,8} ...]], --num_ensembles [{1,8} [{1,8} ...]]
                        the trunk of the network is run multiple times with
                        different random choices for the MSA cluster centers.
                        (1=default, 8=casp14 setting)
  -r MAX_RECYCLES, --max_recycles MAX_RECYCLES
                        controls the maximum number of times the structure is
                        fed back into the neural network for refinement.
                        (default is 3)
  --output_all_cycle    Output the structures of all cycle. If this option is
                        set and max_recycles is 3, the structures of all 1,2,3
                        cycles will be output.
  --tol TOL             tolerance for deciding when to stop (CA-RMS between
                        recycles)
  --is_training         enables the stochastic part of the model (dropout),
                        when coupled with num_samples can be used to 'sample'
                        a diverse set of structures. False (NOT specifying
                        this option) is recommended at first.
  --num_samples NUM_SAMPLES
                        number of random_seeds to try. Default is 1.
  --num_relax {None,Top1,Top5,All}
                        num_relax is 'None' (default), 'Top1', 'Top5' or
                        'All'. Specify how many of the top ranked structures
                        to relax.
  --save_images         save images such as structures and plot
  --show_images         show images
```

### Example

When you run the below command, totally 400 structures are generated by recycling 10 times(all structures while recycling will be output), using 10 models(5 original models and 5 pTM models), with 2 random seeds, and 2 ensemble patterns (1 and 8).

```bash
colabfold-conda/bin/python3.7 runner_af2advanced.py \
-i hoge.fasta \
-o output_dir/hoge \
--max_recycles 10 \
--num_samples 2 \
--model both \
--num_CASP14_models 5 \
--num_pTM_models 5 \
--noranking \
--output_all_cycle \
--num_ensembles 1 8
```

If you have precomputed MSA, add `--msa_method precomputed --precomputed msa.pickle`.

### Output

```bash
model_1_ptm_seed_0_rec_1_ens_1.pdb  # pdb file
model_1_ptm_seed_0_rec_1_ens_1.pickle  # Contains pLDDT, etc.
model_1_ptm_seed_0_rec_2_ens_1.pdb
model_1_ptm_seed_0_rec_2_ens_1.pickle
...
msa.pickle
scores.csv  # Contains pLDDT and pTM for predicted models
msa_coverage.png
```

scores.csv

```txt
,Model,pLDDT,pTMscore,Tolerance,ModelName,Seed,Recycle,Ensemble,Target
0,model_1_ptm_seed_0_rec_10_ens_1,0.8671514626497182,0.754113431050468,0.4492591,model_1_ptm,0,10,1,hoge
1,model_1_ptm_seed_0_rec_10_ens_8,0.8726767323938525,0.7699067933832592,0.58676547,model_1_ptm,0,10,8,hoge
2,model_1_ptm_seed_0_rec_1_ens_1,0.876012646623183,0.7756131184815608,19.154978,model_1_ptm,0,1,1,hoge
```

## Reference paper
- Mirdita M, Schuetze K, Moriwaki Y, Heo L, Ovchinnikov S and Steinegger M. ColabFold - Making protein folding accessible to all. *bioRxiv*, doi: [10.1101/2021.08.15.456425](https://www.biorxiv.org/content/10.1101/2021.08.15.456425v2) (2021)
- John Jumper, Richard Evans, Alexander Pritzel, et al. -  Highly accurate protein structure prediction with AlphaFold. *Nature*, 1–11, doi: [10.1038/s41586-021-03819-2](https://www.nature.com/articles/s41586-021-03819-2) (2021)

## Reference repository

- ColabFold (Commit: [9546d8f](https://github.com/sokrypton/ColabFold/tree/9546d8fbbf77a0d59c9b234486e6e6e1c7765dd4))
- localcolabfold (Commit: [0129393](https://github.com/YoshitakaMo/localcolabfold/tree/0129393473f901d7930f0e2d9d8469a46327bbc8))

## Model Parameters License

The AlphaFold parameters are made available for non-commercial use only, under
the terms of the Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0) license. You can find details at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode