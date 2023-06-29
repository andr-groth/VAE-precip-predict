# Example data

In this folder, we provide example data for the Jupyter notebook examples in the repositories [`examples/`](/examples/) folder. The data is in netCDF format and has been prepared with the help of the CDO scripts, which are available at:

> https://github.com/andr-groth/CDO-scripts

## Data sources

### CMIP6

The data in the [`data/cmip6/historical/`](/data/cmip6/historical/) folder is based on the CMIP6 historical experiment. The different variables are stored in separate subfolders, .i.e, [`data/cmip6/historical/pr/`](/data/cmip6/historical/pr/) for precipitation and `data/cmip6/historical/tos/` for sea-surface temperature.

The original CMIP6 data can be downloaded from the ESGF data portal with the help of the `wget` scripts, which are available in the `raw/` folders of the different subfolders of the variables.

The scripts are taken from the ESGF data portal at:

> https://aims2.llnl.gov/metagrid/search/?project=CMIP6


### Observational data

The observational data in the `data/ersst/` folder is based on [NOAA's Extended Reconstructed SST V5 (ERSSTv5)](https://doi.org/10.1175/jcli-d-16-0836.1) dataset, which is available at:

> https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html

The observational data in the `data/gpcc/` folder is based on [GPCC's Full Data Monthly Product Version 2022](http://dx.doi.org/10.5676/DWD_GPCC/FD_M_V2022_100) dataset, which is available at:

> https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-monthly_v2022_doi_download.html

### Crop data

The data in the `crop/` folder is based on the [GGCMI crop calendar Phase 3](https://doi.org/10.1038/s43016-021-00400-y) dataset, which is available at:

> https://zenodo.org/record/5062513

## Data preparation

### CMIP6

The data preparation of the CMIP6 data involves the following steps:

1. Download data from the ESGF data portal with the help of the `wget` scripts in the `raw/` folders (see above).
2. Merge the data into single files with the help of the `merge.sh` script from the CDO-scripts repository.
3. Create the anomalies, EOFs and PCs with the help of the `prepare_data.sh` script from the CDO-scripts repository.

The configuration of the `prepare_data.sh` script is stored in `anom.cfg` files in the variable subfolders, .i.e, `data/cmip6/historical/pr/anom.cfg` for precipitation and `data/cmip6/historical/tos/anom.cfg` for sea-surface temperature.

### Observational data

The data preparation of the observational data involves the following steps:

#### ERSSTv5
1. Download source data from the data portal (see above).
2. Create the anomalies and project the anomalies onto the CMIP6 EOFs in the `data/cmip6/historical/tos/pcs/eofs.nc` file with the help of the `prepare_data2.sh` script from the CDO-scripts repository.

The configuration of the `prepare_data2.sh` script is stored `data/ersst/anom.cfg`.

#### GPCC
1. Download source data from the data portal (see above).
2. Create the anomalies and project the anomalies onto the CMIP6 EOFs in the `data/cmip6/historical/pr/pcs/eofs.nc` file with the help of the `prepare_data2.sh` script from the CDO-scripts repository.

The configuration of the `prepare_data2.sh` script is stored `data/gpcc/anom.cfg`.

### Crop data

The data has been projected onto the CMIP6 EOFs with the help of the CDO tool `cdo remapcon`.
