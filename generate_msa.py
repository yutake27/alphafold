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

msa_method = "mmseqs2"  # @param ["mmseqs2","single_sequence"]

precomputed = None

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
