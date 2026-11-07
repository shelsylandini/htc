# HTC-Twin

HTC-Twin provides research components for a gut-joint mechanistic model, offline reinforcement learning, exercise-prescription optimization, and off-policy evaluation in knee osteoarthritis.

## Installation

```bash
pip install -e ".[dev]"
```

## Data and execution

Patient records and trained model weights are not included. A dataset-specific loader is not implemented; the Python `OfflineTrainer` requires a caller-provided `LoggedDataset`. The `distil`, `taste`, and `pour` command-line workflows cannot run until data integration is implemented. Reported study results have not been reproduced using this repository.

The quantum optimization code computes a statevector on classical hardware; it does not run on a quantum processor. The mechanistic model and feature mapping require independent validation before their outputs can support research conclusions.

`mash` contains data structures and batching; `ferment` contains the mechanistic model; `still` contains the policy networks; `cut` contains objectives; `blend` contains prescription optimization; `cask` contains training and control; `tasting` and `nose` contain evaluation utilities. Configurations are in `recipes/`.

Potential data sources and access information are listed in `dataset_sources.txt`.

```bash
pytest
```

This is research software and has not been validated for clinical use.
