# Rough model description

This document outlines the model at a high level: well enough to be understood in a general sense, but only for quick reference—the final report should be the ground truth of any model implementation details.



## Stages of the model

### Stage 1. Identification and reproduction

The first stage identifies the model to be used as a starting point. For this, Manore et al. (2015)'s[^1] model was chosen due to its novel network-patch architecture for vector-borne disease spread. This stage also reproduces the model and validates its correctness against the original model results by replicating experiments and comparing outputs.

### Stage 2. Extension and parameterisation

The second stage extends the baseline model. While the model from Manore et al. is useful from a pure modelling perspective, Manore et al. simulate a fictional scenario with no grounding in real geographical data, despite their model accommodating for geographical and environmental heterogeneity.

To ground the model in a real-world setting, I use data collected across three studies from the Mondulkiri Province in rural Cambodia. These data are used to create a scenario in the model that is characteristic of a real-world setting for vector-borne disease spread, not necessarily to restrict the model to a specific setting.

The data from the Cambodia studies are used to parameterise the Manore et al. network-patch model and extend the underlying agent-based model to conform to the setting. Details of how these data informed the model parameters and extensions are provided later in this document.

### Stage 3. Implementation of behavioural change theories

The final stage of the project involves implementing and overlaying behavioural change theories that impact whether or not agents choose to use preventive measures (thus modelling their preventive behaviours).



## Model description

==TODO==. This is basic—just overview of model.

### ODD

- purpose and patterns
- entities, state variables, and scales
- process overview and scheduling
- design concepts
	- basic principles
	- emergence
	- adaptation
	- objectives
	- learning
	- prediction
	- sensing
	- interaction
	- stochasiticity
	- collectives
	- observation
- intialisation
- input data
- submodels



## Extensions to the model & parameterisation

#### Deriving $K_v$:

We know the biting rate of mosquitoes across regions on one subject from data, which is $b_h=b/N_h$, or the average number of bites per host per unit time (2 hrs). As such, we have, for a given patch:
$$
b_h=b/N_h=\frac{\sigma_vN_v\sigma_hN_h}{\sigma_vN_v+\sigma_hN_h}/N_h
$$
we know $\sigma_v,\sigma_h,N_h$. Rearranging for $N_v$​, we have:
$$
N_v=-\frac{b_h\sigma_hN_h}{\sigma_v(b_h-\sigma_h)}
$$
we know $b,\sigma_v,\sigma_h,N_h$. Solving for $N_v$​, we have:
$$
N_v=-\frac{b\sigma_h}{\sigma_v(b/N_h-\sigma_h)}
$$






---

[^1]: https://doi.org/10.1080/17513758.2015.1005698