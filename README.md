# ISIC 2018 Task 3 — Class-Incremental Learning

This repository contains the implementation and results for a class-incremental skin-lesion recognition experiment using the **ISIC 2018 Task 3** dataset. It compares a classical CNN-based model with a hybrid quantum-classical model across sequential learning tasks and rehearsal-memory settings.

## Repository structure

```text
isic2018_task3/
├── results/
│   ├── classical_seed_7597/
│   └── quantum_seed_7597/
├── .gitignore
├── isic2018_classical.py
├── isic2018_quantum.py
├── README.md
└── requirements.txt
```

### Files and folders

| Path | Description |
|---|---|
| `isic2018_classical.py` | Runs the classical continual-learning experiment. |
| `isic2018_quantum.py` | Runs the hybrid quantum-classical continual-learning experiment. |
| `results/classical_seed_7597/` | Output files for the classical experiment using seed `7597`. |
| `results/quantum_seed_7597/` | Output files for the quantum experiment using seed `7597`. |
| `requirements.txt` | Python package dependencies. |
| `.gitignore` | Files and directories excluded from Git. |

## Experiment overview

The experiment studies how well models retain previously learned skin-lesion classes while learning new classes sequentially.

- Dataset: ISIC 2018 Task 3
- Random seed: `7597`
- Learning setting: class-incremental learning
- Task sequence: six classes split across three sequential tasks
- Omitted class: MEL
- Main metric: balanced accuracy, to account for class imbalance
- Rehearsal settings: `0%`, `5%`, `10%`, `20%`, and `100%`
- Evaluation: task-by-task balanced accuracy (R-matrix), final average balanced accuracy, average forgetting, and parameter drift

The classical model uses a CNN-based classification approach. The hybrid model uses CNN image features followed by a compact variational quantum circuit (VQC) classification head. The quantum circuit is simulated on a classical computer; this repository does not demonstrate quantum hardware speed-up.

> **Note:** The exact task-to-class mapping and preprocessing details should be checked in the experiment scripts and reported alongside any results.

## Dataset

Download and prepare the ISIC 2018 Task 3 dataset according to the dataset's official terms and the preprocessing expected by the scripts.

This repository does not include the dataset. Keep the image files and any patient-related or sensitive data out of Git unless you have explicit permission to distribute them.

Before running the experiments, confirm that:
1. The dataset is available locally.
2. The data paths in the scripts point to the correct dataset location.
3. The expected image and label files are present.
4. The train/test split and class sequence match the experiment configuration.

## Installation

Use a Python environment appropriate for the packages listed in `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the experiments

Run the classical experiment:

```bash
python isic2018_classical.py
```

Run the hybrid quantum-classical experiment:

```bash
python isic2018_quantum.py
```

The scripts use the configured dataset paths and experiment settings. Review those settings before execution, especially the dataset location, task sequence, rehearsal ratio, and output directory.

## Results

Experiment outputs are saved under the corresponding seed-specific directory in `results/`.

- Classical outputs: `results/classical_seed_7597/`
- Hybrid quantum-classical outputs: `results/quantum_seed_7597/`

When reporting results, include the rehearsal ratio and seed so that results from different configurations are not confused. Use the full R-matrix to show performance on each task after each learning stage, rather than reporting only the final average.

## Reproducibility

The experiments use random seed `7597` for the recorded run. Reproducibility can also depend on the software environment, hardware, dataset version, data split, and nondeterministic operations. Record these details when rerunning or extending the experiments.

## Limitations

- The dataset is class-imbalanced.
- Results may depend on the selected task order and random seed.
- Continual learning can cause forgetting of previously learned classes.
- The quantum circuit is evaluated using classical simulation.
- A hybrid model's performance does not by itself establish a quantum advantage.


