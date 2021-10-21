#!/bin/bash

# Create execution environment for alphafold
# 1. Download model parameter
# 2. Download chemical property
# 3. Create miniconda environment
# 4. Apply OpenMM patch

ALPHAFOLD_DIR=`pwd`
SOURCE_URL="https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar"
PARAMS_DIR="${ALPHAFOLD_DIR}/alphafold/data/params"

# Downloading alphafold model parameter files
if [ ! -d ${PARAMS_DIR} ];then
    echo "Downloading AlphaFold2 trained parameters..."
    mkdir -p ${PARAMS_DIR}
    curl -fL ${SOURCE_URL} | tar x -C ${PARAMS_DIR}
fi

# Downloading stereo_chemical_props.txt from https://git.scicore.unibas.ch/schwede/openstructure
streo_chemical_props_path=${ALPHAFOLD_DIR}/alphafold/common
if [ ! -e ${stereo_chemical_props_path} ]; then
    echo "Downloading stereo_chemical_props.txt..."
    wget -q https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt --no-check-certificate
    mv stereo_chemical_props.txt ${ALPHAFOLD_DIR}/alphafold/common
fi

# Install Miniconda3 for Linux if execution flag is set
echo "Installing Miniconda3 for Linux..."
pushd ${ALPHAFOLD_DIR}
wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${ALPHAFOLD_DIR}/conda
rm Miniconda3-latest-Linux-x86_64.sh
popd

echo "Creating conda environments with python3.7 as ${ALPHAFOLD_DIR}/colabfold-conda"
. "${ALPHAFOLD_DIR}/conda/etc/profile.d/conda.sh"
export PATH="${ALPHAFOLD_DIR}/conda/condabin:${PATH}"
conda create -p ${ALPHAFOLD_DIR}/colabfold-conda python=3.7 -y
conda activate ${ALPHAFOLD_DIR}/colabfold-conda
conda update -y conda

echo "Installing conda-forge packages"
conda install -c conda-forge python=3.7 cudnn==8.2.1.32 cudatoolkit==11.1.1 openmm==7.5.1 pdbfixer -y
conda install -c bioconda hmmer==3.3.2 hhsuite==3.3.0 -y
echo "Installing alphafold dependencies by pip"
python3.7 -m pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow-gpu==2.5.0
python3.7 -m pip install jupyter matplotlib py3Dmol tqdm
python3.7 -m pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Apply OpenMM patch.
echo "Applying OpenMM patch..."
(cd ${ALPHAFOLD_DIR}/colabfold-conda/lib/python3.7/site-packages/ && patch -p0 < ${ALPHAFOLD_DIR}/docker/openmm.patch)