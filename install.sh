#!/bin/bash

# Exit immediately on any error
set -e

# Environment name
ENV_NAME=echi_recipe

# Detect Conda base path
CONDA_BASE=$(conda info --base)

# Make sure Conda is initialized in this shell
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Optional: run `conda init` once to fix future shell sessions
# Comment this out if you don't want to touch user shell config
conda init bash >/dev/null 2>&1 || true

# Create the conda environment (if it doesn't already exist)
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    conda create --name "$ENV_NAME" python=3.11 -y
fi

# Activate the environment
conda activate "$ENV_NAME"

# Install dependencies
if [[ -f environment.yaml ]]; then
    conda env update --name "$ENV_NAME" --file environment.yaml --prune
else
    echo "ERROR: environment.yaml not found"
    exit 1
fi

# Install Versa from GitHub
echo "Installing Versa Speech Audio Toolkit ..."
python -m pip install git+https://github.com/wavlab-speech/versa.git#egg=versa-speech-audio-toolkit --no-build-isolation

# Install the pysepm package from GitHub
echo "Installing pysepm ..."
python -m pip install git+https://github.com/ftshijt/pysepm.git --no-build-isolation

# Set PYTHONPATH
echo "export PYTHONPATH=\$PWD/src:\$PYTHONPATH" >>~/.bashrc
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Run NISQA setup
if [[ -f external/versa/tools/setup_nisqa.sh ]]; then
    (
        cd external/versa/tools || exit 1
        bash setup_nisqa.sh
    )

else
    echo "ERROR: NISQA setup script not found at external/versa/tools/setup_nisqa.sh"
    exit 1
fi

echo "✅ Environment '$ENV_NAME' is set up and ready to use!"
