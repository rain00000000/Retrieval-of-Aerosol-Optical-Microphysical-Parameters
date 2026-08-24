# Retrieval of Aerosol Optical/Microphysical Parameters

This repository provides the key source code and datasets for the paper:
> "Enhanced retrieval of aerosol optical/microphysical parameters for Himawari-8 geostationary satellite measurements with data-driven deep learning method"

## Overview

We propose a Deep Belief Network (DBN) based method to retrieve aerosol parameters (AOD, SSA, fine/coarse-mode AOD) from Himawari-8 geostationary satellite measurements.

## Files

| File | Description |
|------|-------------|
| `DBN.py` | DBN model definition, including pretraining and fine-tuning |
| `RBM.py` | Restricted Boltzmann Machine layer implementation |
| `AOD1.zip`, `AOD2.zip` | Aerosol Optical Depth data |
| `SSA.zip` | Single Scattering Albedo data |
| `SDA1.zip`, `SDA2.zip`, `SDA3.zip` | Fine-mode AOD (FAOD), Coarse-mode AOD (CAOD), and related SDA products |
| `model.zip` | Pre-trained model weights |

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, scikit-learn, matplotlib
