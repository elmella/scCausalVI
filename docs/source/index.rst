scCausalVI
======================================
scCausalVI is a causality-aware generative model for analyzing perturbational single-cell RNA sequencing data. The model addresses a fundamental challenge in single-cell analysis: disentangling intrinsic cellular heterogeneity from treatment-induced effects, in particular for
case-control study.
By incorporating structural causal models with deep learning, scCausalVI:

- Learns disentangled and interpretable latent representations that separate baseline cellular states from treatment effects
- Models cell-state-specific responses to explore differential response pattern
- Enables in silico perturbation to predict cellular states under alternative experimental conditions
- Integrates multi-source data while distinguishing batch effects from biological signals (baseline states and treatment effects)
- Identifies treatment-responsive populations and characterizes molecular signatures of susceptibility and resistance to disease

The framework supports comprehensive downstream analyses, including clustering, visualization, differential expression analyses,
and cross-condition prediction, providing researchers with tools to investigate cellular heterogeneity and treatment responses at single-cell resolution.

.. image:: _static/overview.png
   :alt: Overview


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   installation
   tutorial
   api

.. include:: README.md
   :parser: myst_parser.sphinx_
