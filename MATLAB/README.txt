README for the MATLAB code of "Some Pleasant Sequence-Space Arithmetic in Continuous Time" by Adrien Bilal and Shlok Goyal

Note the code is written assuming two idiosyncratic states. One can extend this to multiple states by having
the first dimension be all the gridpoints for the endogenous state and the second dimension all the gridpoints
for the exogenous state (so if there are two endogenous states, store all possible pairs of each as one endogenous state).

Main File Descriptions:

HA_All.m: This is a self-contained file for the HA model.
    It solves the steady state and then goes through each step of our algorithm to calculate the Jacobians.
    It then plots the Jacobians and IRFs.

HANK_All.m: This is a self-contained file for the HANK model.
    It solves the steady state and then goes through each step of our algorithm to calculate the Jacobians.
    It then plots the Jacobian and IRFs we show in the paper.

Helper Functions:

AxisFonts.m: Sets graphical parameters for plots.

IRFs_fn: Function to plot impulse responses.

Jacobian_Plots: Function to plot the columns of Jacobians.

markov_rouwenhorst: Function to create the productivity grid via Rouwenhorst's method.

Folder Descriptions:

Figures: Where all plots and the CSV files for runtimes are stored.

