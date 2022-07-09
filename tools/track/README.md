## Tracking benchmark on V2XSIM

Here we implements the sort algorithm as our benchmark trackers and use the detection results obtained from [here](../det) to evaluate.

## Preparation
- Download V2XSIM datasets from our [website](https://ai4ce.github.io/V2X-Sim/index.html)
- Prepare tracking ground truth:
```bash
make create_data
```
- You might want to consult `./Makefile` for all the arguments you can pass in

Create seqmaps (required by the SORT codebase):
```base
make create_seqmaps
```


## Evaluation

Run a tracker:
```bash
make sort
```
- You might want to consult `./Makefile` for all the arguments you can pass in


Evaluate tracking results:

```bash
make eval
```
- Results will be stored in `./logs` directory.  
- You might want to consult `./Makefile` for all the arguments you can pass in



## Results

|  **Method**   | **MOTA**      | **MOTP**      | **HOTA**      | **DetA**      | **AssA**      | **DetRe**     | **DetPr**     | **AssRe**     | **AssPr**     | **LocA**      |
| :-----------: | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|  Lower-bound  | 39.25 (+3.83) | 85.04 (-1.05) | 34.29 (+1.44) | 40.46 (+2.51) | 30.44 (+0.65) | 43.26 (+2.57) | 79.31 (+0.12) | 40.40 (+1.11) | 53.26 (-0.43) | 86.87 (-0.94) |
|   When2com    | 39.72 (+3.86) | 85.13 (-1.40) | 33.61 (+2.61) | 39.27 (+4.02) | 30.11 (+1.10) | 41.29 (+4.84) | 85.24 (-2.81) | 39.91 (+1.48) | 53.72 (-0.65) | 87.11 (-1.19) |
| When2com_warp | 40.13 (+2.76) | 83.87 (-0.08) | 34.16 (+0.74) | 40.89 (+1.20) | 30.20 (+0.19) | 42.88 (+1.82) | 79.76 (+0.43) | 40.28 (+0.29) | 52.98 (+0.14) | 85.99 (+0.01) |
|    Who2com    | 39.72 (+3.85) | 85.13 (+1.68) | 33.61 (+2.61) | 39.27 (+4.01) | 30.12 (+1.10) | 41.29 (+4.88) | 82.23 (-2.82) | 39.91 (+1.48) | 53.72 (-0.65) | 87.11 (-1.19) |
| Who2com_warp  | 40.13 (+2.77) | 83.87 (-0.09) | 34.16 (+0.74) | 40.28 (+1.80) | 30.20 (+0.19) | 42.87 (+1.83) | 79.76 (+0.44) | 40.28 (+0.29) | 52.98 (+0.14) | 86.00 (+0.01) |
|    V2VNet     | 58.11 (+4.34) | 85.23 (-0.08) | 42.41 (+1.64) | 55.83 (+3.52) | 33.31 (+0.45) | 58.79 (+3.93) | 84.68 (-0.05) | 44.10 (+0.77) | 54.04 (-0.49) | 87.05 (+0.05) |
|   DiscoNet    | 58.51 (+4.26) | 85.53 (+0.32) | 42.34 (+2.29) | 56.72 (+3.80) | 32.65 (+1.31) | 60.08 (+3.78) | 84.52 (+1.02) | 43.63 (+1.62) | 53.88 (-0.23) | 87.33 (+0.21) |
|  Upper-bound  | 61.31 (+5.58) | 86.00 (-1.08) | 43.12 (+2.62) | 58.77 (+4.43) | 32.55 (+1.51) | 61.22 (+5.35) | 86.65 (-1.36) | 43.25 (+1.77) | 53.56 (-0.11) | 87.68 (-0.86) |


