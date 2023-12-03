# Modernizing the ATLAS Statistical Analysis: Implementing HistFitter Strategies with pyhf in Supersymmetry search

## Abstract

This contribution signifies a shift in ATLAS statistical data analysis by implementing traditional fit strategies utilizing the `pyhf` library, alongside the `cabinetry` library. Leveraging a toy Supersymmetry search analysis, three fit strategies inspired by the HistFitter framework are implemented. The "background-only fit," "model-dependent signal fit," and "model-independent signal fit" strategies show the adaptability of `pyhf`, liberating the analysis from dependence on traditional ROOT-based tools. In addition to enhancing clarity regarding the statistical model itself, this implementation signifies a broader shift towards contemporary standards in data analytics.

## Getting started

The notebook [SUSY_pyfits.ipynb](https://gitlab.cern.ch/ekourlit/pyhf2023-atlas-susy-fits/-/blob/master/SUSY_pyfits.ipynb) which contains the main body of this contribution is self-explanatory. All the data required are provided in `histograms/`. Utility functions are defined in `tools.py`.

### Binder

Click the button below to run the notebook online via Binder:

[![Binder](https://binderhub.ssl-hep.org/badge_logo.svg)](https://binderhub.ssl-hep.org/v2/git/https%3A%2F%2Fgitlab.cern.ch%2Fekourlit%2Fpyhf2023-atlas-susy-fits/HEAD?labpath=SUSY_pyfits.ipynb)