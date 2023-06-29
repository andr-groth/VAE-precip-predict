# Example data

In this folder, we provide example data for the Jupyter notebook examples in the `examples/` folder. The data is in netCDF format and has been prepared with the help of the CDO scripts, which are available at:

> https://github.com/andr-groth/cdo-scripts

## Data sources

### CMIP6

The data in the `cmip6/historical/` folder is based on the CMIP6 historical experiment. The different variables are stored in separate subfolders, .i.e, `cmip6/historical/pr/` for precipitation and `cmip6/historical/tos/` for sea-surface temperature.

The original CMIP6 data can be downloaded from the ESGF data portal with the help of the `wget` scripts, which are available in the `raw/` folders of the different subfolders.

The scripts are taken from the ESGF data portal at:

> https://aims2.llnl.gov/metagrid/search/?project=CMIP6


### Observational data

The observational data in the `ersst/` folder is based on _NOAA's Extended Reconstructed SST V5_ (ERSSTv5) dataset, which is available at:

> https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html

The observational data in the `gpcc` folder is based on the _GPCC Full Data Monthly Product Version 2022_ dataset, which is available at:

> https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-monthly_v2022_doi_download.html

## Data preparation

### CMIP6

The data preparation of the CMIP6 data involves the following steps:

1. Download data from the ESGF data portal with the help of the `wget` scripts in the `raw/` folders (see above).
2. Merge the data into single files with the help of the `merge.sh` script from the CDO-scripts repository.
3. Create the anomalies, EOFs and PCs with the help of the `prepare_data.sh` script from the CDO-scripts repository.

Note:
    The configuration of the `prepare_data.sh` script is stored in `anom.cfg` files in the variable subfolders, .i.e, `cmip6/historical/pr/anom.cfg` for precipitation and `cmip6/historical/tos/anom.cfg` for sea-surface temperature.

### Observational data

The data preparation of the observational data involves the following steps:

#### ERSSTv5
1. Download source data from the data portal (see above).
2. Create the anomalies and project the anomalies onto the CMIP6 EOFs in the `cmip6/historical/tos/pcs/eofs.nc` file with the help of the `prepare_data2.sh` script from the CDO-scripts repository.

The configuration of the `prepare_data2.sh` script is stored `ersst/anom.cfg`.

#### GPCC
1. Download source data from the data portal (see above).
2. Create the anomalies and project the anomalies onto the CMIP6 EOFs in the `cmip6/historical/pr/pcs/eofs.nc` file with the help of the `prepare_data2.sh` script from the CDO-scripts repository.

The configuration of the `prepare_data2.sh` script is stored `gpcc/anom.cfg`.
