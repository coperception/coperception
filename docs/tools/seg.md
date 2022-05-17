# Segmentation


We implement lowerbound, upperbound, when2com, who2com, V2VNet as our benchmark segmentation methods. Please see more details in our paper.  
You can find and configure all the arguments for the program in the `tools/seg/Makefile` file.


## Preparation

Choose one of our supported datasets, for example, [V2X-Sim](/datasets/v2x_sim), and perform its data preparation steps based on our documentation.

## Training

Train benchmark segmentation methods:  

- Lowerbound
```bash
# lowerbound
make train_bound bound=lowerbound

# lowerbound with out cross road (RSU)
make train_bound_nc bound=lowerbound
```

- Upperbound
```bash
# upperbound
make train_bound bound=upperbound

# lowerbound with out cross road (RSU) data
make train_bound_nc bound=upperbound
```

- V2VNet
```bash
# V2V
make train com=v2v

# V2V with no cross road (RSU) data
make train_nc com=v2v
```

- DiscoNet
```bash
# DiscoNet
make train_disco

# DiscoNet with no cross road (RSU) data
make train_disco_nc
```

- When2com
```bash
# When2com
make train com=when2com

# When2com with no cross road (RSU) data
make train_nc com=when2com
```

- When2com_warp
```bash
# When2com_warp
make train_warp com=when2com

# When2com_warp with no cross road (RSU) data
make train_warp_nc com=when2com
```

- Note: Who2com is trained the same way as When2com. They only differ in inference.

## Evaluation

Evaluate benchmark segmentation methods
```bash
# lowerbound
make test_bound bound=lowerbound

# lowerbound with no cross road (RSU) data
make test_bound_nc bound=lowerbound
```

- Upperbound
```bash
# upperbound
make test_bound bound=upperbound

# upperbound with no cross road (RSU) data
make test_bound_nc bound=upperbound
```
- V2VNet
```bash
# V2V
make test com=v2v

# V2V with no cross road (RSU) data
make test_nc com=v2v
```

- DiscoNet
```bash
# DiscoNet
make test com=disco

# DiscoNet with no cross road (RSU) data
make test_nc com=disco
```

- When2com
```bash
# When2com
make test_w inference=activated

# When2com with no cross road (RSU) data
make test_w_nc inference=activated
```

- When2com_warp
```bash
# When2com_warp
make test_warp inference=activated

# When2com_warp with no cross road (RSU) data
make test_warp_nc inference=activated
```

- Who2com
```bash
# Who2com
make test_w inference=argmax_test

# Who2com with no cross road (RSU) data
make test_w_nc inference=argmax_test
```

- Who2com_warp
```bash
# Who2com
make test_warp inference=argmax_test

# Who2com with no cross road (RSU) data
make test_w_nc inference=argmax_test
```
## Results

| **Method**    | **Vehicle**   | **Sidewalk**  | **Terrain**   | **Road**      | **Building**  | **Pedestrian** | **Vegetation** |   **mIoU**    |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------- | -------------- | :-----------: |
| Lower-bound   | 57.48 (+2.97) | 83.43 (+3.16) | 81.70 (+2.83) | 89.73 (+1.42) | 80.93 (+2.82) | 25.98 (+0.35)  | 74.29 (+4.00)  | 70.51 (+2.51) |
| When2com      | 58.44 (+3.56) | 65.46 (+4.05) | 62.29 (+3.30) | 80.79 (+2.08) | 61.31 (+3.59) | 27.28 (-1.70)  | 60.04 (+3.72)  | 59.37 (+2.66) |
| When2com_warp | 58.64 (+3.15) | 62.21 (+8.58) | 60.24 (+6.12) | 79.57 (+3.69) | 59.29 (+6.30) | 26.36 (-2.28)  | 59.43 (+6.19)  | 57.96 (+4.54) |
| Who2com       | 58.63 (+3.37) | 62.57 (+6.94) | 59.62 (+5.97) | 80.08 (+2.80) | 59.00 (+5.91) | 26.60 (-1.02)  | 59.09 (+4.67)  | 57.94 (+4.09) |
| Who2com_warp  | 58.64 (+3.15) | 62.21 (+8.58) | 60.24 (+6.12) | 79.57 (+3.69) | 59.29 (+6.29) | 26.36 (-2.28)  | 59.43 (+6.19)  | 57.96 (+4.53) |
| V2VNet        | 67.83 (+4.69) | 84.43 (+3.87) | 81.83 (+4.62) | 91.76 (+2.36) | 75.95 (+4.15) | 26.70 (+2.45)  | 74.95 (+4.30)  | 72.49 (+3.78) |
| DiscoNet      | 66.88 (+3.30) | 81.80 (+1.21) | 79.68 (+0.56) | 90.80 (+1.01) | 80.16 (-1.99) | 28.79 (+0.19)  | 72.92 (-0.04)  | 71.58 (+0.61) |
| Upper-bound   | 73.13 (+4.26) | 84.60 (+7.31) | 83.72 (+3.08) | 92.68 (+1.59) | 81.45 (+3.95) | 35.23 (+2.03)  | 77.34 (+2.78)  | 75.45 (+3.14) |

