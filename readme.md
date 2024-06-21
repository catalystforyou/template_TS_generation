# Hypothesized Transition State Configuration Generation

## Environment
Please install the following software before running the script:
```
conda install -c conda-forge rdkit
pip install scipy tqdm
```
Please also make sure that the `xtb` command is available in your system.

## Step 1
Please prepare your input files in the following format:
- SH.gjf: Pre-optimized geometry of the thiol molecule.
- Radical.gjf: Pre-optimized geometry of the radical.
(The optimization could be done using Gaussian or other software, please contact me if you need to enrich the types of input files.)

## Step 2
Please run the following command to generate the transition state configuration:
```
python gen_TS.py
```
The script will generate two hypothesized TS configurations named as `combined_0.mol` and `combined_1.mol`, alongwith an auxiliary file named as `fixed.inp`.
Please note that the generated TS configurations are only placed by the template and need to be further optimized.

## Step 3
Please run the following command to optimize the hypothesized TS configurations:
```
xtb combined_0.mol --gbsa toluene --opt --input fixed.inp --uhf 1
```
The optimized TS configuration will be saved as `xtbopt.mol`, please rename it as `pre_TS_0.mol` for further analysis(`mv xtbopt.mol pre_TS_0.mol`).
After that, please run the following command to optimize the other hypothesized TS configuration:
```
xtb combined_1.mol --gbsa toluene --opt --input fixed.inp --uhf 1
```
The optimized TS configuration will be saved as `xtbopt.mol`, please rename it as `pre_TS_1.mol` for further analysis(`mv xtbopt.mol pre_TS_1.mol`).

## Step 4
Please use GaussView or other software to visualize the optimized TS configurations and further analyze the results.