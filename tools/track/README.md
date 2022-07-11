# Tracking benchmark on V2XSIM
Here we implements the sort algorithm as our benchmark trackers and use the detection results obtained from [here](../det) to evaluate.

## Preparation
- Download V2XSIM datasets from our [website](https://ai4ce.github.io/V2X-Sim/index.html)
- Prepare tracking ground truth:
```bash
make create_data
```
You might want to consult `./Makefile` for all the arguments you can pass in.  
For example, the target for `create_data` is:
```bash
create_data:
	python create_data_com.py --root $(original_data_path) --data $(det_data_path)/$(split) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file)
```
You should at least set `original_data_path` to the path of V2X-Sim dataset on your machine, and `det_data_path` to the path of the preprocessed detection dataset.  
You can set the variables at the top of `Makefile`, or you can pass them in as arguments.  
For other arguments, please see the comments in `Makefile`.  

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

|   **Method**   | **MOTA**      | **MOTP**      | **HOTA**      | **DetA**      | **AssA**      | **DetRe**     | **DetPr**     | **AssRe**     | **AssPr**     | **LocA**      |
| :------------: | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|  Lower-bound   | 35.72 (-3.87) | 84.16 (-0.74) | 34.27 (-1.68) | 33.64 (-3.24) | 36.18 (-0.06) | 35.07 (-3.54) | 82.49 (+0.96) | 46.70 (+0.23) | 58.72 (+0.10) | 86.43 (+0.38) |
| Co-lower-bound | 21.53 (+0.58) | 85.76 (+0.15) | 39.16 (-0.71) | 41.14 (-0.93) | 38.18 (-0.62) | 59.54 (-2.52) | 54.68 (+0.79) | 50.92 (-0.65) | 55.78 (+0.84) | 87.64 (+0.38) |
|    When2com    | 29.48 (+2.45) | 86.10 (-2.79) | 30.94 (+1.01) | 27.90 (+2.04) | 35.33 (+0.06) | 28.67 (+2.58) | 86.11 (-4.81) | 46.30 (-0.15) | 59.20 (-0.36) | 87.98 (-1.98) |
|   When2com*    | 30.17 (+1.43) | 84.95 (-1.44) | 31.34 (+0.43) | 29.11 (+1.05) | 35.42 (+0.21) | 30.28 (+1.32) | 83.81 (+0.29) | 46.65 (-0.29) | 58.61 (+0.18) | 86.14 (+0.17) |
|    Who2com     | 29.48 (+2.46) | 86.10 (-2.79) | 30.94 (+1.01) | 27.90 (+2.04) | 35.33 (+0.06) | 28.67 (+2.58) | 86.11 (-4.81) | 46.30 (-0.15) | 59.20 (-0.36) | 87.98 (-1.98) |
|    Who2com*    | 30.17 (+1.43) | 84.95 (-1.44) | 31.34 (+0.43) | 29.11 (+1.06) | 35.42 (+0.21) | 30.28 (+1.33) | 83.81 (+0.29) | 46.65 (-0.29) | 58.61 (+0.81) | 86.14 (+0.17) |
|     V2VNet     | 55.29 (+2.29) | 85.21 (-0.53) | 43.68 (+0.91) | 50.71 (+1.93) | 38.76 (+0.24) | 53.40 (+2.51) | 84.45 (-1.07) | 50.22 (+0.53) | 58.50 (-0.07) | 87.22 (+0.38) |
|    DiscoNet    | 56.69 (+2.26) | 86.23 (-0.41) | 44.76 (+1.09) | 52.41 (+2.18) | 39.25 (+1.11) | 54.87 (+2.58) | 86.29 (-0.95) | 50.86 (+1.02) | 58.94 (-0.15) | 88.07 (+0.34) |
|  Upper-bound   | 58.00 (+3.92) | 85.61 (+0.25) | 44.83 (+4.24) | 52.94 (+4.24) | 38.95 (-0.75) | 55.07 (+4.68) | 86.54 (-0.30) | 50.35 (-0.86) | 58.71 (+0.15) | 87.48 (+0.06) |