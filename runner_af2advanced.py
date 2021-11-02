# Copyright (c) 2021 Yoshitaka Moriwaki
# Copied from https://github.com/YoshitakaMo/localcolabfold

# %%
# command-line arguments

import argparse
import datetime
import json
import os
import pickle
import platform
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from tqdm import tqdm

import colabfold as cf
import colabfold_alphafold as cf_af

parser = argparse.ArgumentParser(description="Runner script that can take command-line arguments")
parser.add_argument("-i", "--input", help="Path to a FASTA file. Required.", required=True)
parser.add_argument("-o", "--output_dir", default="", type=str,
                    help="Path to a directory that will store the results. "
                    "The default name is 'prediction_<hash>'. ")
parser.add_argument("-ho", "--homooligomer", default="1", type=str,
                    help="homooligomer: Define number of copies in a homo-oligomeric assembly. "
                    "For example, sequence:ABC:DEF, homooligomer: 2:1, "
                    "the first protein ABC will be modeled as a omodimer (2 copies) and second DEF a monomer (1 copy). Default is 1.")
parser.add_argument("-m", "--msa_method", default="mmseqs2", type=str, choices=["mmseqs2", "single_sequence", "precomputed"],
                    help="Options to generate MSA."
                    "mmseqs2 - FAST method from ColabFold (default) "
                    "single_sequence - use single sequence input."
                    "precomputed - specify 'msa.pickle' file generated previously if you have."
                    "Default is 'mmseqs2'.")
parser.add_argument("--precomputed", default=None, type=str,
                    help="Specify the file path of a precomputed pickled msa from previous run. "
                    )
parser.add_argument("-p", "--pair_mode", default="unpaired", choices=["unpaired", "unpaired+paired", "paired"],
                    help="Experimental option for protein complexes. "
                    "Pairing currently only supported for proteins in same operon (prokaryotic genomes). "
                    "unpaired - generate seperate MSA for each protein. (default) "
                    "unpaired+paired - attempt to pair sequences from the same operon within the genome. "
                    "paired - only use sequences that were sucessfully paired. "
                    "Default is 'unpaired'.")
parser.add_argument("-pc", "--pair_cov", default=50, type=int,
                    help="Options to prefilter each MSA before pairing. It might help if there are any paralogs in the complex. "
                    "prefilter each MSA to minimum coverage with query (%%) before pairing. "
                    "Default is 50.")
parser.add_argument("-pq", "--pair_qid", default=20, type=int,
                    help="Options to prefilter each MSA before pairing. It might help if there are any paralogs in the complex. "
                    "prefilter each MSA to minimum sequence identity with query (%%) before pairing. "
                    "Default is 20.")
parser.add_argument("-b", "--rank_by", default="pLDDT", type=str, choices=["pLDDT", "pTMscore"],
                    help="specify metric to use for ranking models (For protein-protein complexes, we recommend pTMscore). "
                    "Default is 'pLDDT'.")
parser.add_argument("--noranking", action="store_true", help="Ranking output structures. If True, the output file name contains a ranking.")
parser.add_argument("-t", "--use_turbo", action='store_true',
                    help="introduces a few modifications (compile once, swap params, adjust max_msa) to speedup and reduce memory requirements. "
                    "Disable for default behavior.")
parser.add_argument("-mm", "--max_msa", default="512:1024", type=str,
                    help="max_msa defines: max_msa_clusters:max_extra_msa number of sequences to use. "
                    "This option ignored if use_turbo is disabled. Default is '512:1024'.")
parser.add_argument("--num_CASP14_models", default=5, type=int,
                    help="specify how many CASP14 model (normal model) params to try. (Default is 5)")
parser.add_argument("--num_pTM_models", default=5, type=int,
                    help="specify how many pTM model params to try. (Default is 5)")
parser.add_argument("--model", type=str, choices=["CASP14", "pTM", "both"], default="CASP14",
                    help="Model to use. CASP14 is a normal model. Both uses both models. Number of models to be used is read from num_CASP14_models and num_pTM_models respectively. (Default is CASP14)")
parser.add_argument("-e", "--num_ensembles", nargs='*', default=['1'], choices=['1', '8'],
                    help="the trunk of the network is run multiple times with different random choices for the MSA cluster centers. "
                    "(1=default, 8=casp14 setting)")
parser.add_argument("-r", "--max_recycles", default=3, type=int,
                    help="controls the maximum number of times the structure is fed back into the neural network for refinement. (default is 3)")
parser.add_argument("--output_all_cycle", action="store_true", help="Output the structures of all cycle. If this option is set and max_recycles is 3, the structures of all 1,2,3 cycles will be output.")
parser.add_argument("--tol", default=0, type=float,
                    help="tolerance for deciding when to stop (CA-RMS between recycles)")
parser.add_argument("--is_training", action='store_true',
                    help="enables the stochastic part of the model (dropout), when coupled with num_samples can be used to 'sample' a diverse set of structures. False (NOT specifying this option) is recommended at first.")
parser.add_argument("--num_samples", default=1, type=int, help="number of random_seeds to try. Default is 1.")
parser.add_argument("--num_relax", default="None", choices=["None", "Top1", "Top5", "All"],
                    help="num_relax is 'None' (default), 'Top1', 'Top5' or 'All'. Specify how many of the top ranked structures to relax.")
parser.add_argument("--save_images", action='store_true', help="save images such as structures and plot")
parser.add_argument("--show_images", action="store_true", default=False, help="show images")
args = parser.parse_args()

# Time
start_datetime = datetime.datetime.now()
print(start_datetime)

# command-line arguments
# Check your OS for localcolabfold
pf = platform.system()
if pf == 'Windows':
    print('ColabFold on Windows')
elif pf == 'Darwin':
    print('ColabFold on Mac')
    device = "cpu"
elif pf == 'Linux':
    print('ColabFold on Linux')
    device = "gpu"
# %%
# python code of AlphaFold2_advanced.ipynb
tf.config.set_visible_devices([], 'GPU')


TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

try:
    from google.colab import files
    IN_COLAB = True
except:
    IN_COLAB = False

# Set flag for images
save_images = True if args.save_images else False
show_images = True if args.show_images else False  # @param {type:"boolean"}

# %%
# define sequence
# --read sequence from input file--


def readfastafile(fastafile):
    records = list(SeqIO.parse(fastafile, "fasta"))
    if(len(records) != 1):
        raise ValueError('Input FASTA file must have a single ID/sequence.')
    else:
        return records[0].id, records[0].seq


print("Input ID: {}".format(readfastafile(args.input)[0]))
print("Input Sequence: {}".format(readfastafile(args.input)[1]))
sequence = str(readfastafile(args.input)[1])
# --read sequence from input file--
jobname = "test"  # @param {type:"string"}
homooligomer = args.homooligomer  # @param {type:"string"}

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

# prediction directory
# --set the output directory from command-line arguments
if args.output_dir != "":
    output_dir = args.output_dir
else:
    output_dir = Path(args.input).stem

I = cf_af.prep_inputs(sequence, jobname, homooligomer, output_dir, clean=IN_COLAB)

# MSA generation
print()
msa_prep_start_datetime = datetime.datetime.now()
print(msa_prep_start_datetime, 'MSA preparation')

msa_method = args.msa_method  # @param ["mmseqs2","single_sequence"]

if msa_method == "precomputed":
    if args.precomputed is None:
        raise ValueError("ERROR: `--precomputed` undefined. "
                         "You must specify the file path of previously generated 'msa.pickle' if you set '--msa_method precomputed'.")
    else:
        precomputed = args.precomputed
        print("Use precomputed msa.pickle: {}".format(precomputed))
else:
    precomputed = args.precomputed

add_custom_msa = False  # @param {type:"boolean"}
msa_format = "fas"  # @param ["fas","a2m","a3m","sto","psi","clu"]

# --set the output directory from command-line arguments
pair_mode = args.pair_mode  # @param ["unpaired","unpaired+paired","paired"] {type:"string"}
pair_cov = args.pair_cov  # @param [0,25,50,75,90] {type:"raw"}
pair_qid = args.pair_qid  # @param [0,15,20,30,40,50] {type:"raw"}
# --set the output directory from command-line arguments

# --- Search against genetic databases ---

I = cf_af.prep_msa(I, msa_method, add_custom_msa, msa_format, pair_mode, pair_cov, pair_qid,
                   hhfilter_loc="colabfold-conda/bin/hhfilter", precomputed=precomputed, TMP_DIR=output_dir)
mod_I = I

if len(I["msas"][0]) > 1:
    plt_msas = cf.plot_msas(I["msas"], I["ori_sequence"], return_plt=True)
    plt_msas.savefig(os.path.join(I["output_dir"], "msa_coverage.png"), bbox_inches='tight', dpi=200)
    # plt.show()
# %%
trim = ""  # @param {type:"string"}
trim_inverse = False  # @param {type:"boolean"}
cov = 0  # @param [0,25,50,75,90,95] {type:"raw"}
qid = 0  # @param [0,15,20,25,30,40,50] {type:"raw"}

mod_I = cf_af.prep_filter(I, trim, trim_inverse, cov, qid)

if I["msas"] != mod_I["msas"]:
    plt.figure(figsize=(16, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.title("Sequence coverage (Before)")
    cf.plot_msas(I["msas"], I["ori_sequence"], return_plt=False)
    plt.subplot(1, 2, 2)
    plt.title("Sequence coverage (After)")
    cf.plot_msas(mod_I["msas"], mod_I["ori_sequence"], return_plt=False)
    plt.savefig(os.path.join(I["output_dir"], "msa_coverage.filtered.png"), bbox_inches='tight', dpi=200)
    if show_images:
        plt.show()

msa_prep_end_datetime = datetime.datetime.now()
print(msa_prep_end_datetime, 'Finish preparing MSA')
print('MSA preparation time', msa_prep_end_datetime - msa_prep_start_datetime)

# %%
# @title run alphafold
print()
alphafold_start_datetime = datetime.datetime.now()
print(alphafold_start_datetime, 'Run alphafold')

# --------set parameters from command-line arguments--------
num_relax = args.num_relax
rank_by = args.rank_by

use_turbo = True if args.use_turbo else False
max_msa = args.max_msa
# --------set parameters from command-line arguments--------

max_msa_clusters, max_extra_msa = [int(x) for x in max_msa.split(":")]

# --------set parameters from command-line arguments--------
model = args.model
num_CASP14_models = args.num_CASP14_models
num_pTM_models = args.num_pTM_models
max_recycles = args.max_recycles
tol = args.tol
is_training = True if args.is_training else False
num_samples = args.num_samples
ranking = not args.noranking
# --------set parameters from command-line arguments--------

subsample_msa = True  # @param {type:"boolean"}

# Set models
models_CASP14 = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5'][: num_CASP14_models]
models_pTM = ['model_1_ptm', 'model_2_ptm', 'model_3_ptm', 'model_4_ptm', 'model_5_ptm'][: num_pTM_models]
use_ptm = False
if model == "CASP14":
    model_names = models_CASP14
elif model == "pTM":
    model_names = models_pTM
    use_ptm = True
else:
    model_names = models_CASP14 + models_pTM

if not use_ptm and rank_by == "pTMscore":
    print("WARNING: models will be ranked by pLDDT, 'use_ptm' is needed to compute pTMscore")
    rank_by = "pLDDT"

# prep input features
feature_dict = cf_af.prep_feats(mod_I, clean=IN_COLAB)
Ls_plot = feature_dict["Ls"]

# prep model options
opt = {"N": len(feature_dict["msa"]),
       "L": len(feature_dict["residue_index"]),
       "use_turbo": use_turbo,
       "max_recycles": max_recycles,
       "tol": tol,
       "max_msa_clusters": max_msa_clusters,
       "max_extra_msa": max_extra_msa,
       "is_training": is_training}


runner = None

###########################
# run alphafold
###########################
outs, structure_names = {}, []
num_ensembles = list(map(int, args.num_ensembles))
for num_ensemble in num_ensembles:
    print('Ensemble', num_ensemble)
    opt['num_ensemble'] = num_ensemble
    outs_, structure_names_ = cf_af.run_alphafold(feature_dict, opt, runner, model_names, num_samples,
                                                  subsample_msa, show_images=show_images,
                                                  output_all_cycle=args.output_all_cycle)
    outs.update(outs_)
    structure_names.extend(structure_names_)

if ranking:  # rank output structures
    structure_names = [structure_names[i] for i in np.argsort([outs[x][rank_by] for x in structure_names])[::-1]]
    # Write out the prediction
    for n, key in enumerate(structure_names):
        prefix = f"rank_{n+1}_{key}"
        pred_output_path = os.path.join(feature_dict["output_dir"], f'{prefix}.pdb')
        if show_images:
            fig = cf.plot_protein(outs[key]["unrelaxed_protein"], Ls=feature_dict["Ls"], dpi=200)
            if save_images:
                plt.savefig(os.path.join(feature_dict["output_dir"], f'{prefix}.png'), bbox_inches='tight')
            plt.close(fig)
        from alphafold.common import protein
        pdb_lines = protein.to_pdb(outs[key]["unrelaxed_protein"])
        with open(pred_output_path, 'w') as f:
            f.write(pdb_lines)

        tmp_pdb_path = os.path.join(feature_dict["output_dir"], f'{key}.pdb')
        if os.path.isfile(tmp_pdb_path):
            os.remove(tmp_pdb_path)

    ############################################################
    print(f"model rank based on {rank_by}")
    for n, key in enumerate(structure_names):
        print(f"rank_{n+1}_{key} {rank_by}:{outs[key][rank_by]:.2f}")

alphafold_end_datetime = datetime.datetime.now()
print(alphafold_end_datetime, 'Finish running alphafold')
print('Alphafold running time', alphafold_end_datetime - alphafold_start_datetime)
print()

# %%
# @title Refine structures with Amber-Relax (Optional)

# --------set parameters from command-line arguments--------
num_relax = args.num_relax
# --------set parameters from command-line arguments--------

if num_relax == "None":
    num_relax = 0
elif num_relax == "Top1":
    num_relax = 1
elif num_relax == "Top5":
    num_relax = 5
else:
    num_relax = len(model_names) * num_samples

if num_relax > 0:
    if "relax" not in dir():
        # add conda environment to path
        sys.path.append('./colabfold-conda/lib/python3.7/site-packages')

        # import libraries
        from alphafold.relax import relax, utils

    with tqdm(total=num_relax, bar_format=TQDM_BAR_FORMAT) as pbar:
        pbar.set_description(f'AMBER relaxation')
        for n, key in enumerate(structure_names):
            if n < num_relax:
                prefix = f"rank_{n+1}_{key}" if ranking else key
                pred_output_path = os.path.join(I["output_dir"], f'{prefix}_relaxed.pdb')
                if not os.path.isfile(pred_output_path):
                    amber_relaxer = relax.AmberRelaxation(
                        max_iterations=0,
                        tolerance=2.39,
                        stiffness=10.0,
                        exclude_residues=[],
                        max_outer_iterations=20)
                    relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=outs[key]["unrelaxed_protein"])
                    with open(pred_output_path, 'w') as f:
                        f.write(relaxed_pdb_lines)
                pbar.update(n=1)
# %%
# @title Display 3D structure {run: "auto"}
rank_num = 1  # @param ["1", "2", "3", "4", "5"] {type:"raw"}
color = "lDDT"  # @param ["chain", "lDDT", "rainbow"]
show_sidechains = False  # @param {type:"boolean"}
show_mainchains = False  # @param {type:"boolean"}

if ranking:
    key = structure_names[rank_num-1]
    prefix = f"rank_{rank_num}_{key}"
    pred_output_path = os.path.join(I["output_dir"], f'{prefix}_relaxed.pdb')
    if not os.path.isfile(pred_output_path):
        pred_output_path = os.path.join(I["output_dir"], f'{prefix}.pdb')

    cf.show_pdb(pred_output_path, show_sidechains, show_mainchains, color, Ls=Ls_plot).show()
    if color == "lDDT":
        if show_images:
            cf.plot_plddt_legend().show()
    if use_ptm:
        plt_confidence =  cf.plot_confidence(outs[key]["plddt"], outs[key]["pae"], Ls=Ls_plot)
    else:
        plt_confidence = cf.plot_confidence(outs[key]["plddt"], Ls=Ls_plot)
    if show_images:
        plt_confidence.show()
# %%
# @title Extra outputs
dpi = 300  # @param {type:"integer"}
save_to_txt = False  # @param {type:"boolean"}
save_pae_json = False  # @param {type:"boolean"}
save_score = True

if save_images:
    if use_ptm:
        print("predicted alignment error")
        cf.plot_paes([outs[k]["pae"] for k in structure_names], Ls=Ls_plot, dpi=dpi)
        plt.savefig(os.path.join(I["output_dir"], f'predicted_alignment_error.png'),
                    bbox_inches='tight', dpi=np.maximum(200, dpi))
        if show_images:
            plt.show()

    print("predicted contacts")
    cf.plot_adjs([outs[k]["adj"] for k in structure_names], Ls=Ls_plot, dpi=dpi)
    plt.savefig(os.path.join(I["output_dir"], f'predicted_contacts.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))
    if show_images:
        plt.show()

    print("predicted distogram")
    cf.plot_dists([outs[k]["dists"] for k in structure_names], Ls=Ls_plot, dpi=dpi)
    plt.savefig(os.path.join(I["output_dir"], f'predicted_distogram.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))
    if show_images:
        plt.show()

    print("predicted LDDT")
    cf.plot_plddts([outs[k]["plddt"] for k in structure_names], Ls=Ls_plot, dpi=dpi)
    plt.savefig(os.path.join(I["output_dir"], f'predicted_LDDT.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))
    if show_images:
        plt.show()


def do_save_to_txt(filename, adj, dists, sequence):
    adj = np.asarray(adj)
    dists = np.asarray(dists)
    L = len(adj)
    with open(filename, "w") as out:
        out.write("i\tj\taa_i\taa_j\tp(cbcb<8)\tmaxdistbin\n")
        for i in range(L):
            for j in range(i+1, L):
                if dists[i][j] < 21.68 or adj[i][j] >= 0.001:
                    line = f"{i}\t{j}\t{sequence[i]}\t{sequence[j]}\t{adj[i][j]:.3f}"
                    line += f"\t>{dists[i][j]:.2f}" if dists[i][j] == 21.6875 else f"\t{dists[i][j]:.2f}"
                    out.write(f"{line}\n")


def read_output_info_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        o = pickle.load(f)
        plddt = o["pLDDT"] / 100
        ptm = o["pTMscore"] if "pTMscore" in o else None
        tol = o["tol"]
    return plddt, ptm, tol


# Save plddt and pTM to csv
def save_score_to_csv(filename):
    score_list = []
    pattern = re.compile(r'^.*(model_[1-5](_ptm)?)_seed_(\d)_rec_(\d)_ens_(\d)$')
    for out_pkl in Path(output_dir).glob("model_*.pickle"):
        model = out_pkl.stem
        model_name, _, seed, recycle, ensemble = pattern.search(model).groups()
        plddt, ptm, tol = read_output_info_from_pkl(out_pkl)
        score_list.append((model, plddt, ptm, tol, model_name, seed, recycle, ensemble))
    df = pd.DataFrame(score_list, columns=["Model", "pLDDT", "pTMscore", "Tolerance",
                                           "ModelName", "Seed", "Recycle", "Ensemble"])
    target_name = Path(args.input).stem
    df["Target"] = target_name
    df = df.sort_values('Model').reset_index()
    df.to_csv(filename)


if save_score:
    save_score_to_csv(os.path.join(output_dir, "scores.csv"))


for n, key in enumerate(structure_names):
    if save_to_txt:
        if ranking:
            txt_filename = os.path.join(I["output_dir"], f'rank_{n+1}_{key}.raw.txt')
        else:
            txt_filename = os.path.join(I["output_dir"], f'{key}.raw.txt')
        do_save_to_txt(txt_filename,
                       outs[key]["adj"],
                       outs[key]["dists"],
                       mod_I["full_sequence"])

    if use_ptm and save_pae_json:
        pae = outs[key]["pae"]
        max_pae = pae.max()
        # Save pLDDT and predicted aligned error (if it exists)
        pae_output_path = os.path.join(I["output_dir"], f'rank_{n+1}_{key}_pae.json')
        # Save predicted aligned error in the same format as the AF EMBL DB
        rounded_errors = np.round(np.asarray(pae), decimals=1)
        indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
        indices_1 = indices[0].flatten().tolist()
        indices_2 = indices[1].flatten().tolist()
        pae_data = json.dumps([{
            'residue1': indices_1,
            'residue2': indices_2,
            'distance': rounded_errors.flatten().tolist(),
            'max_predicted_aligned_error': max_pae.item()
        }],
            indent=None,
            separators=(',', ':'))
        with open(pae_output_path, 'w') as f:
            f.write(pae_data)

# Finish
finish_datetime = datetime.datetime.now()
print(finish_datetime, 'Finish')
print('Total time', finish_datetime - start_datetime)
