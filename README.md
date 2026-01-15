# Multi-Chain EIP-1559 Instability Research Artifact

## Overview

This repository contains the implementation and simulation code for analyzing EIP-1559 transaction fee mechanisms in multi-chain blockchain systems. 

Critically, it implements the multi-chain fee market with coupled chains. Local and cross-chain workflow react to the fees, and the fees are updated based on the EIP-1559 rule per chain. 

To reproduce each figure in the manuscript, we provide concrete simulation scripts to generate the raw data, and the diagram scripts to paint the raw data and deliver the figures. The simulation methodology is also justified in the manuscript.

Moreover, to enable non-simulation configuration tuning, we include a simple prediction model to map configurations to the amplification factors.

## Structure

- abstract-fee-and-load. Scripts and data for the motivation section, including empirical data for fee volatility of Polkadot, and instability of Cosmos.
- core. Implementation of the fee market and the negative-feedback rule.
- nn-model. Prediction model for the amplification factors, including the training code and the model.
- simulation. simulation scripts which iterate critical coupling-related parameters, such as cross-chain ratio, latency, elasticity, and update rate. We validate in this repository whether there exists actual amplification (fast-simulations), how the delayed coupling causes safety margin erosions, and how current architectures can retain responsiveness while maintaining stability (time-consuming-simulations).

## Quick Start

Each figure is reproducible via the scripts presented in the 'simulations' repository.

1. Fast-simulations

This directory contains two sets of scripts, for validating the amplification of the delayed defense channel and the delayed attack channel.

- correlation-kappa-def-and-self-delay. It measures whether and to what extent the delayed coupling amplifies the negative feedback of EIP-1559 and causes instability.
- appendix-correlation-kappa-att-and-delay-attack. It measures whether and to what extent the delayed positive feedback is amplified.

2. Time-consuming-simulations

This direction contains 13 sets of scripts, for quantifying the safety erosion caused by the delayed coupling, deriving the maximum update rate for realistic configurations, validating the stability boundary (which confirms that our delay-folding approximation is correct and sound), and validating the prediction capability of the non-simulation framework. In the appendix, we validate whether the amplification of the delayed attack channel exists, and checks whether it threatens stability. Moreover, we explore the impact of different maximum latency and different chain numbers on the stability.

- stable-region-with-fixed-delta. It measures the stable region of the cross-chain ratio and elasticity. It demonstrates that the multi-chain architecture can only tolerate a low elasticity when cross-chain workflow is extensive.
- delta-safe-under-fixed-elasticity. It measures the critical stability metric, the safety margin of the update rate delta. It shows that the multi-chain architecture suffers from a severe safety erosion in terms of the update rate with the delayed economic coupling.
- delta-max-under-realistic-configurations. It derives the maximum update rate for the six real-world blockchains in Polkadot and Cosmos systems.
- stability-boundary-validation. It validates whether our derived theoretical stability boundary is correct, by first calculating the amplification factor, and then comparing the stability boundary's mathematical recommendation (with the calculated amplification factor) against the simulation outcome.
- stability-boundary-prediction. It validates how accurate the non-simulation nn-model can predict or in other words, map the configuration parameters to the amplification factor, derives the prediction using this predicted value of kappa, and compares the prediction with the simulation outcome.
- predicted-delta-max. It iterates the update rate to determine the maximum update rate for the six chains using the prediction model. During the iteration, for each update rate, it calculates the amplification factor, and calculate the R_i and G_i. When the calculated R_i and G_i does not meet the stability boundary, it stops.
- delta-max-for-random-settings. It first generates random configurations, including the cross-chain ratio, elasticity, update rate, latency, etc. It then use this parameter set to predict the maximum update rate, similar to the process of 'predicted-delta-max'.

--- 
The followings are the scripts for figures presented in the Appendix.

- appendix-stability-boundary-for-oc-with-dmax-three, appendix-stability-boundary-for-oc-with-dmax-seven. These two sets of scripts quantify the impact of different dmax on the stability.
- appendix-stability-boundary-for-oc-with-varying-chain-numbers. It quantifies the impact of different chain numbers on the stability.
- appendix-stability-boundary-enhancing-ji-delay. It quantifies whether the delayed attack channel can harm the stability.
- appendix-stability-boundary-enhancing-inst. It quantifies whether the instantaneous attack channel can harm the stability.
- appendix-stability-boundary-enhancing-inst-under-dmax-one. To isolate the delayed defense channel of other chains, we rerun the scripts in appendix-stability-boundary-enhancing-inst and checks whether the channel can harm the stability.


Each directory contains the script for generating the simulation raw data, and we provide the raw data in the .csv files so that people with interests can directly validate our results. To visualize this data, we provide the script for generating the figures and diagrams in the manuscript.

3. Empirical data in the motivation section

We provide empirical data for the fee comparison between Moonbeam and Ethereum, and the load comparison between Osmosis and Ethereum. 
Steps to rerun the empirical analysis: 
- For fee, run fetch_fee.sh.
- For load, use fetch_rpc_hour_windows.py for Osmosis and range_load_extract.py for Ethereum.

Remember to use your own key to rerun the empirical analysis.


Note: All figures and data are reproducible, and we welcome any question regarding the methodology and the artifact.