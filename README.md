# CHiME-9 Task 2 - ECHI Baseline

### Enhancing Conversation to address Hearing Impairment

This repository contains the baseline system for the CHiME-9 challenge, Task 2 ECHI.

> **Quick start?** To get started with the data, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CHiME9-ECHI/CHiME9-ECHI/blob/main/Quickstart.ipynb), or jump to [Getting Started](#getting-started) to begin using this repo.

For detailed information on how to participate in the challenge and for obtaining the datasets, please refer to the [challenge website](https://www.chimechallenge.org/current/task2/index)

---

## About the Challenge

**CHiME-9 Task 2: Enhancing Conversations for Hearing Impairment (ECHI)** addresses the task of trying to separate conversations between known participants from noisy cafeteria-like backgrounds with low latency processing. Solutions to this problem would benefit [huge numbers](https://rnid.org.uk/get-involved/research-and-policy/facts-and-figures/prevalence-of-deafness-and-hearing-loss/) of people with mild hearing loss.

![concept](docs/images/echi_concept.png)

### Key Challenge Features

- **Natural conversation** each of over 30 minutes with up to four people.
- **49 sessions** and **196 accent-diverse speakers**
- **High overlap** both in-conversation and with distractor talkers.
- **Noisy background** simulating varying-loudness cafeterias
- **Moving multi-microphone input** from 4-channel hearing aids and 7-channel Aria glasses
- **Headtracking** for all participants.

### Evaluation Metrics

Systems are evaluated using:

1. **Reference-based**: Quality and intelligibility estimation
2. **Reference-free**: DNN-based metric predictors
3. **Listening tests**: Human-rated intelligibility and quality to rank systems

### Challenge documentation

For detailed information about the challenge, please refer to the main website:

- **[Challenge Overview](https://www.chimechallenge.org/current/task2/index)** - Full challenge description
- **[Data Description](https://www.chimechallenge.org/current/task2/data)** - Dataset structure and access
- **[Baseline System](https://www.chimechallenge.org/current/task2/baseline)** - System architecture and components
- **[Challenge Rules](https://www.chimechallenge.org/current/task2/rules)** - Participation rules and evaluation guidelines

---

## Getting Started

### Sections

1. <a href="#install">Installing the software</a>
2. <a href="#data">Installing the dataset</a>
3. <a href="#stages">Stages</a>
4. <a href="#troubleshooting">Troubleshooting</a>

## <a id="#install">1. Installing the software</a>

Clone this repository from GitHub

```bash
git clone git@github.com:CHiME9-ECHI/CHiME9-ECHI.git
cd CHiME9-ECHI
```

The installation of the necessary tools is detailed in `install.sh`.
We recommend following it step-by-step and adjusting for your system if needed.
The script will build a conda environment called `echi_recipe` and install the
dependencies listed in `environment.yaml`.

When running the system, remember to activate the conda environment and set the
necessary environment variables,

```bash
conda activate echi_recipe
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

To make the `PYTHONPATH` setting persistent across terminal sessions, you can add
 the `export` command to your shell's configuration file (e.g., `~/.bashrc` for
 bash or `~/.zshrc` for zsh).

## <a id="data"> 2. Installing the data </a>

Our dataset is hosted on HuggingFace. To download the dataset, you must first
request access at the
[CHiME9-ECHI Dataset page](https://huggingface.co/datasets/CHiME9-ECHI/CHiME9-ECHI).
Full details of the dataset can be found at that link. Once you have been
granted access, you can download the dataset using the following commands:

```bash
# Navigate to your baseline project directory if you aren't already there
# cd CHiME9-ECHI

# Create a data directory (if it doesn't exist)
mkdir -p data
cd data

# Export your Hugging Face token (replace 'your_actual_token_here'
# with your actual token)
export HF_TOKEN=your_actual_token_here

# Download the development set
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/CHiME9-ECHI/CHiME9-ECHI/resolve/main/data/chime9_echi.dev.v1_0.tar.gz

# Untar the downloaded files
tar -zxovf chime9_echi.dev.v1_0.tar.gz
```

The baseline system expects the dataset to be placed in the `data/chime9_echi`
 directory by default. The paths to various subsets of the data (e.g., training,
 development, evaluation) are defined in the `config/paths.yaml` file. If you
 choose to place the data in a different location, you will need to update
 `config/paths.yaml` accordingly. However, adhering to the default directory
 structure (`data/chime9_echi`) is recommended for ease of use.

Note that `data/chime9_echi` can also be a symbolic link to another location,
allowing flexibility in where the dataset is physically stored.

## <a id="stages">3. Stages</a>

Once your environment is set up, there are three main ways you might want to use this repository:

- Evaluate enhanced signals you've generated
- Run an enhancement system (e.g. run the baseline)
- Train your own model
Choose the workflow below that fits your goal:

✅ **Evaluation:** If you have enhanced signals to evaluate

Given a directory containing all the enhanced files, this script computes all the specified metrics over all sessions. Details can be found on the [evaluation page](docs/evaluation.md).

> **⚠️ WARNING:**
> **Evaluation code should be considered read-only.**
> Any modifications to the evaluation scripts could lead to invalid results.
> If there are any problems which cannot be resolved without editing the code,
> please raise an issue and we will respond accordingly.

🎧 **Enhancement:** If you want to try the baseline enhancement system

Given a system, the enhancement pipeline produces
 audio for each full session and saves it with the correct formatting. Default
 options for passthrough and the baseline are provided, but custom options
 can be added. Details can be found on the
 [enhancement page](docs/enhancement.md).

🧠  **Train:** If you want to train your own system

Prepares speech segments of the dataset and then trains using them.
Details can be found on the [training page](docs/training.md).

## <a id="troubleshooting">4. Troubleshooting</a>

If you encounter issues, here are some common troubleshooting steps:

- **Activate Conda Environment:** Ensure your Conda environment (`echi_recipe`) is
 activated:

  ```bash
  conda activate echi_recipe
  ```

- **Check PYTHONPATH:** Verify that your `PYTHONPATH` environment variable is correctly
 set to include the `src` directory of this project:

  ```bash
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  # (or ensure this is in your .bashrc or equivalent shell startup script)
  echo $PYTHONPATH
  ```

- **Verify Data Paths:** Double-check that the dataset paths in `config/paths.yaml`
 match the actual location of your CHiME-9 ECHi data. The default expected location
 is `data/chime9_echi`.
- **Hydra Log Files:** For detailed error messages and execution logs, inspect the
 Hydra log files. These are typically found in the `exp/<experiment_name>/hydra/`
 directory (e.g., `exp/main/hydra/`). The exact path will be printed at the start
 of a run.
- **Common Python Issues:** Check for common Python package installation problems
 or version conflicts within the Conda environment. Sometimes, reinstalling a
 problematic package can help.
- **Other Issues:** If you have any other problems with this repository that
cannot be fixed, please raise a GitHub Issue and we will do our best to resolve
it for you.
