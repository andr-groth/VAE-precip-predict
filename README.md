# Seasonal-level prediction of climate data with a variational autoencoder

## Overview

The Jupyter notebooks demonstrate the process of training and exploring a Variational Autoencoder (VAE) on precipitation and sea-surface temperature data. The training process is divided into two steps: pre-training on CMIP6 data and transfer learning on observational data.

The framework is based on Groth & Chavez (2023). _submitted_.

## Requirements

1. The Jupyter notebooks requires the VAE package, which is available at:

    https://github.com/andr-groth/VAE-project

2. Sample data used in the notebook is included in the `data/` folder. The data is in netCDF format and has been prepared with the help of the CDO scripts, which are available at:

    https://andr-groth.github.io/cdo-scripts