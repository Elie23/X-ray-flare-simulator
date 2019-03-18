This code generates X-ray lightcurves from user input parameters. Bayesian Blocks are then run on the simulated event list and the flare duration and fluences are printed to the sceen.

Input parameters:
- Quiescence count rate (ct/s)
- Exposure (s)
- Instrument mode (ACIS-S/subarray or ACIS-S/HETG/0th+1st orders)
- Flare duration (s), unabsorbed fluences (10^{37} erg) and center time (s)
- Quiescence count rate


Optional parameters:
- For any instrument mode:
   - Pile-up grade migration parameter (default=1)
   - Flare count rate to unabsorbed luminosity ratio (pile-up corrected and quiescence subtracted) (default: 0.013 ct/10^{34} erg for subarray flare and 0.077 ct/(10^{34} erg for HETG flares)

- For HETG only:
    - Quiescence 0th/1st count rate ratio (default=0.44)
    - Flare 0th/1st count rate (pile-up corrected and quiescence subtracted) (default=1.6)
    
