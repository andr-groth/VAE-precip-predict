# Seasonal-level prediction of climate data with a variational autoencoder

## Overview

Jupyter notebooks for the implementation of a variational autoencoder (VAE) for climate data modeling and prediction.

The Jupyter notebooks demonstrate the process of training and exploring a Variational Autoencoder (VAE) on precipitation and sea-surface temperature data. The training process is divided into two steps: pre-training on CMIP6 data and transfer learning on observational data.

## Requirements

1. The Jupyter notebooks requires the __VAE package__, which is available at:

    > https://github.com/andr-groth/VAE-project

2. Sample data used in the notebook is included in the [`data/`](/data/) folder. The data is in netCDF format and has been prepared with the help of the __CDO scripts__, which are available at:

    > https://andr-groth.github.io/CDO-scripts

    For more information on the data preparation see [`data/README.md`](/data/README.md).

## Usage

This repository contains two Jupyter notebooks for model training and prediction:

1. To train the VAE on CMIP6 data and transfer learn on observational data: [`VAEp_train.ipynb`](VAEp_train.ipynb)

2. To make predictions on observational data: [`VAEp_explore.ipynb`](VAEp_explore.ipynb)

The notebooks use sample data provided in the [`data/`](/data/) folder in this repository.