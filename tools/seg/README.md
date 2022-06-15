## Segmentation benchmark on V2XSIM

We implement lowerbound, upperbound, when2com, who2com, V2VNet as our benchmark segmentation methods. Please see more details in our paper.

## Preparation

- Download V2XSIM datasets from our [website](https://ai4ce.github.io/V2X-Sim/index.html)
- Run the code below to generate preprocessed data

```bash
make_create_data
```
- You might want to consult `./Makefile` for all the arguments you can pass in


## Training

Train benchmark detectors:
- Lowerbound / Upperbound / V2VNet / When2Com
```bash
make train com=[lowerbound/upperbound/v2v/when2com] rsu=[0/1]
```

- DiscoNet
```bash
# DiscoNet
make train_disco

# DiscoNet with no cross road (RSU) data
make train_disco_no_rsu
```

- When2com_warp
```bash
# When2com_warp
make train com=when2com warp_flag=1 rsu=[0/1]
```

- Note: Who2com is trained the same way as When2com. They only differ in inference.

## Evaluation

Evaluate benchmark detectors:

- Lowerbound
```bash
# with RSU
make test com=[lowerbound/upperbound/v2v/when2com/who2com]

# no RSU
make test_no_rsu com=[lowerbound/upperbound/v2v/when2com/who2com]
```

- When2com
```bash
# with RSU
make test com=when2com inference=activated warp_flag=[0/1]

# no RSU
make test_no_rsu com=when2com inference=activated warp_flag=[0/1]
```

- Who2com
```bash
# with RSU
make test com=who2com inference=argmax_test warp_flag=[0/1]

# no RSU
make test_no_rsu com=who2com inference=argmax_test warp_flag=[0/1]
```
## Results

|   **Method**   |  **Vehicle**  | **Sidewalk**  |  **Terrain**  |    **Road**    | **Building**  | **Pedestrian** | **Vegetation** |   **mIoU**    |
| :------------: | :-----------: | :-----------: | :-----------: | :------------: | :-----------: | :------------: | :------------: | :-----------: |
|  Lower-bound   | 45.93 (+2.22) | 42.39 (-2.75) | 47.03 (+0.20) | 65.76 (-1.27)  | 25.38 (-1.89) | 20.59 (-3.09)  | 35.83 (+0.66)  | 36.64 (-0.87) |
| Co-lower-bound | 47.67 (+2.43) | 48.79 (-1.41) | 50.92 (+0.85) | 70.00 (-0.65)  | 25.26 (+0.17) | 10.78 (-1.77)  | 39.46 (+2.69)  | 38.38 (+0.46) |
|    When2com    | 47.87 (+2.34) | 33.73 (-0.95) | 33.65 (-0.50) | 58.05 (+0.06)  | 30.16 (-0.44) | 20.14 (-1.46)  | 38.24 (-1.08)  | 34.49 (-0.52) |
|   When2com*    | 47.74 (+1.23) | 33.60 (-0.40) | 35.81 (+1.05) | 56.75  (+0.48) | 26.11 (-0.92) | 19.16 (+0.04)  | 39.64 (-2.55)  | 33.81 (-0.47) |
|    Who2com     | 47.87 (+2.34) | 33.73 (-0.95) | 33.65 (-0.50) | 58.05 (+0.06)  | 30.16 (-0.44) | 20.14 (-1.46)  | 38.24 (-1.08)  | 34.49 (-0.52) |
|    Who2com*    | 47.74 (+1.23) | 33.60 (-0.40) | 35.81 (+1.05) | 56.75 (+0.48)  | 26.11 (-0.92) | 19.16 (+0.04)  | 39.64 (-2.55)  | 33.81 (-0.47) |
|     V2VNet     | 58.35 (+3.21) | 48.85 (-5.22) | 47.49 (+1.73) | 69.81 (-0.42)  | 29.18 (-1.24) | 22.38 (-0.27)  | 41.25 (+1.46)  | 41.17 (-0.53) |
|    DiscoNet    | 55.84 (+1.89) | 47.88 (-3.03) | 49.19 (-1.60) | 68.03 (-1.50)  | 31.76 (+0.44) | 22.66 (-2.09)  | 42.81 (+0.49)  | 41.34 (-0.97) |
|  Upper-bound   | 64.09 (+5.34) | 41.34 (+2.42) | 48.20 (+0.74) | 67.05 (+2.04)  | 29.07 (+0.74) | 31.54 (+3.15)  | 45.04 (+0.70)  | 42.29 (+1.98) |

