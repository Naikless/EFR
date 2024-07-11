# EFR (Expansion Flow Reactor)

This module allows to conduct quasi-one-dimensional simulations of one-dimensional detonations. 
It uses [CalTech's Shock and Detonation Toolbox](https://shepherd.caltech.edu/EDL/PublicResources/sdt/) to calculate the detonation properties for a specific mixture at defined initial conditions. 
After calculation of the detonation's ZND profile, a Method of Characterictics (MoC) solver is employed to obtain the flow field downstream of the detonation wave, taking into account viscous losses through a convective loss terms based on Reynold's analogy.

The resulting flow fields are then used to conduct chemical reactor network (CRN) studies that allow to investigate the formation of combustion pollutants over time and space.
