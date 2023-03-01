# Spatial Vine Copulas

A project for the Current Research Seminar WS22/23 by Christian Soraruf and Stefan Solarski at LMU.

## Contents

The main file is a Jupyter **notebook** "[Spatial Vine Copulas](https://github.com/SSolarski/spatialvinecopula/blob/537b0648c91f0338795c0c41cff939ca783a0fa7/Spatial_Vine_Copulas.ipynb)". It contains all of our outputs and visualizations.


The three main classes DataSets, SpatialCopula and SpatialVineCopula are located in the [spvinecopulib.py](https://github.com/SSolarski/spatialvinecopula/blob/537b0648c91f0338795c0c41cff939ca783a0fa7/functions/spvinecopulib.py) file in the **functions** folder.

The main example dataset [meuse](https://rsbivand.github.io/sp/reference/meuse.html) can be loaded in the notebook using the [skg](https://pypi.org/project/scikit-gstat/) package.

## Setup

Clone the repository and use [pip](https://pip.pypa.io/en/stable/), or another package manager, to install the requirements.

```bash
git clone https://github.com/SSolarski/spatialvinecopula.git
cd spatialvinecopula
pip install -r requirements.txt
```

Necessary packages are given in "requirements.txt".

The notebook can be run as is. (we used Python v3.8.8).

## Optional

The alternative datasets from the SIC2004 exercise are saved as .csv files in the [sic2004data_01](https://github.com/SSolarski/spatialvinecopula/blob/537b0648c91f0338795c0c41cff939ca783a0fa7/sic2004data_01) folder.

If you wish to run the algorithm on this dataset please uncomment the notebook cell which loads the optional dataset.

## Acknowledgment

This work was based on ideas presented in the paper "[Modelling skewed spatial random fields through the spatial vine copula](https://www.researchgate.net/publication/260011614_Modelling_skewed_spatial_random_fields_through_the_spatial_vine_copula)" by Benjamin Gräler.

Our code is an attempt of recreating the functionalities of the R library [spcopula](https://github.com/BenGraeler/spcopula) by the same author.
