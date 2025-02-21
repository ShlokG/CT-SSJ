README for the Python code of "Some Pleasant Sequence-Space Arithmetic in Continuous Time" by Adrien Bilal and Shlok Goyal

For those looking to understand the algorithm, we recommend starting with the files "HA_All.py" and "HANK_All.py"

Note the code is written assuming two idiosyncratic states. One can extend this to multiple states by having
the first dimension be all the gridpoints for the exogenous state and the second dimension all the gridpoints
for the endogenous state (so if there are two endogenous states, store all possible pairs of each as one endogenous state).

File Descriptions:

HA_All.py: This is a self-contained file for the HA model.
    It solves the steady state and then goes through each step of our algorithm to calculate the Jacobians.
    It then plots the Jacobians and IRFs.

HANK_All.py: This is a self-contained file for the HANK model.
    It solves the steady state and then goes through each step of our algorithm to calculate the Jacobians.
    It then plots the Jacobian and IRFs we show in the paper (Figure 4).

"Model.py": Creates functions necessary to run the HA and HANK models, passing in the model-specific matrices/functions.

"Parameters.py": Functions to return the parameters for the HA and HANK models.

"Steady_State.py": Functions to get the steady state of either the HA or HANK model using either
    Endogenous Gridpoint Method (EGM) or the implicit method.

"Jacobian_Helpers.py": Functions used to calculate Jacobians using our method, following Appendix F.
	Documentation for these functions are in Jacobian_Helpers.html

"HA_Figures.py": Produces the Jacobian and IRF figures for the HA model we show in the paper (Figures 1 and 2) as well as the consumption Jacobians.
    Also produces the .tex files for parameters and steady states (Tables 1 and 2).

"HA_Runtime.py": Calculates runtimes for the steady state and each step of the Jacobian algorithm.
    Does so in 3 cases: discrete time, continuous time with steady state solved via EGM and continuous time with steady state solved via implicit method.
    Produces the runtime plots for the HA model (Figure 3 and 6a as well as Table 5).

"HANK_Figures.py": Produces the Jacobian and IRF figures for the HANK model we show in the paper (Figure 4).
    Also produces the .tex files for parameters and steady states (Tables 3 and 4).

"HANK_Runtime.py": Same as "HA_Runtime.py" but for the HANK model. Produces Figures 5, 6b, and Table 6.

"Runtime_Plot_Fns.py": Functions to produce the runtime plots. Called by "HA_Runtime.py" and "HANK_Runtime.py"

"HA_DT.py": Functions used to calculate the steady state of the HA model in discrete time.
    Also calculates the runtimes of the IRF calculation step in discrete time in the HA model.

"HANK_Helpers.py": Functions to calculate the General Equilibrium Jacobians and IRFs of the HANK model in continuous time
	given the partial equilibrium Jacobians calculated using functions in Jacobian_Helpers.py.
    Also includes functions to calculate each step of the discrete-time algorithm and IRFs for the HANK model in discrete time.

Folder Descriptions:

Figures: Where all plots and the CSV and .tex files for runtimes are stored.
Storage: Parameters for the HANK model and the discrete-time version of the model.
toolkit: Folder from Auclert et al. (2021) that stores utility functions and used for the discrete-time calculations for the HA model.
sequence_jacobian: Folder from Auclert et al. (2024) used for the discrete-time calculations for the HANK model.

