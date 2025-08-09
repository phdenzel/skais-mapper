# Welcome to `skais-mapper`

A framework for generating deep-learning SKA radio telescope &
cosmological hydrodynamical simulation data.


## Table Of Contents

- [Install](install.md)
- [Usage](usage.md)
- [API Reference](reference.md)


`skais-mapper` is a collection of tools for generating, plotting, and
pre-processing hydrodynamics simulation (image) data for
state-of-the-art generative AI models.

It is fully compatible with SPH data from the AREPO simulator, in
particular the [IllustrisTNG suite](https://www.tng-project.org/data/).

It provides utility routines to fetch isolated halos from simulations
snapshots and various raytracing algorithms for 2D column density
projections of these halos and its galaxies. Although the package is
mostly built on python, the raytracing module also includes some C
extensions for the intensive computation (building and visualizing
datasets). The framework generates HDF5 files with image datasets of
various galactic properties, such as dark matter, star, or gas column
density distributions.
