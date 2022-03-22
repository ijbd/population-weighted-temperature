## Introduction

This repository contains hourly, population-weighted temperature series for balancing authorities in the U.S. The analysis relies on two datasets:

1. Temperature data from [NASA MERRA](https://gmao.gsfc.nasa.gov/reanalysis/MERRA/)
2. Population density data from [SEDAC](https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11)

## Dataset Descriptions

**NASA MERRA**
- Years: 2016, 2017, 2018, 2019
- Variables: Ambient temperature ("T2M")

**SEDAC**
- Years: 2015, 2020
- Variables: Population density adjusted for UN WPP country totals

## Output

For each balancing authority, four timeseries files are generated.


File Suffix | Description
--- | ---
temperature-2020-pop.csv | Spatial average of temperature weighted by 2020 population
temperature-2015-pop.csv | Spatial average of temperature weighted by 2015 population
centroid-temperature.csv | Temperature of center-most coordinate.
unweighted-temperature.csv | Unweighted spatial average of temperature.








