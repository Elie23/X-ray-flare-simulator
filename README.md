This code generates X-ray lightcurves from user input parameters. Bayesian Blocks are then run on the simulated event list and the flare duration and fluences are printed to the sceen. The light curve is also plotted. Please note that the bins are 300 s and each point is the middle of a bin, such that there are no points exactly on the edge of a given plot.

To run the code, open the notebook and run it. ModifiedNCP_PRIORxbblocks.py contains the Bayesian blocks implementation (a modified version of Peter William's implementation; https://newton.cx/~peter/2013/05/bayesian-blocks-analysis-in-python/. The txt file contains 10 runs of ncp-prior vs N for a fixed p0=0.05 which are used to calibrate ncp_prior. 

Input parameters: (directly in the notebook)
- Quiescence count rate (ct/s)
- Exposure (s)
- Instrument mode (ACIS-S/subarray or ACIS-S/HETG/0th+1st orders)
- Flare duration (s), unabsorbed fluences (10^{37} erg) and center time (s)


Optional parameters:
- For any instrument mode:
   - Pile-up grade migration parameter (default=1)
   - Flare count rate to unabsorbed luminosity ratio (pile-up corrected and quiescence subtracted) (default: 0.013 ct/10^{34} erg for subarray flare and 0.077 ct/(10^{34} erg for HETG flares)

- For HETG only:
    - Quiescence 0th/1st count rate ratio (default=0.44)
    - Flare 0th/1st count rate (pile-up corrected and quiescence subtracted) (default=1.6)
    
