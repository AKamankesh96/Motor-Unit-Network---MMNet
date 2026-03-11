# Motor Unit Mode Network (MMNet)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18895984.svg)](https://doi.org/10.5281/zenodo.18895984)

This repository provides an implementation of the **Motor-unit Mode Network (MMNet)** for extracting low-dimensional motor-unit modes from motor-unit discharge-rate signals.

MMNet models coordinated activity across multiple motor units using a variational autoencoder architecture to identify latent patterns of neural drive underlying muscle activation.

---

## Repository Contents

- `mmnet.py` – main script for training the MMNet model
- `mmnet_demo.ipynb` – simple notebook demonstrating how to run the model and inspect latent modes

---

## Input Data

MMNet operates on **motor unit discharge-rate signals** obtained from high-density surface EMG (HD-sEMG).

Typical preprocessing pipeline:

1. Record **HD-sEMG signals**
2. Decompose EMG signals to obtain **motor unit spike trains**
3. Convert spike trains into **continuous discharge-rate signals** using smoothing
4. Provide the discharge-rate signals as input to MMNet

The model expects a MATLAB `.mat` file containing a matrix of motor unit
discharge-rate signals (default variable name: `concatenated_data`) with shape: [n_mus, n_samples]

where:

- `n_mus` = number of motor units  
- `n_samples` = number of time samples  

Each row corresponds to one motor unit discharge-rate signal.

---

## Example Command

```bash
python scripts/mmnet.py \
--mat_path All_MUs_S01s35bef.mat \
--latent_dim 4 \
--out_dir outputs_S01 \
--save_full_recon \
--save_full_latent \
--save_config

```

## Citation

If you use MMNet in your research, please cite:

Kamankesh, A. (2026).  
**Motor-Unit Mode Network (MMNet)**.  
Zenodo. https://doi.org/10.5281/zenodo.18895984

BibTeX:

```bibtex
@software{kamankesh_mmnet_2026,
  author  = {Kamankesh, Alireza},
  title   = {Motor-Unit Mode Network (MMNet)},
  year    = {2026},
  doi     = {10.5281/zenodo.18895984},
  url     = {https://github.com/AKamankesh96/Motor-Unit-Network---MMNet}
}
