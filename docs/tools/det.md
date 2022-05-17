# Detection

We implement lowerbound, upperbound, when2com, who2com, V2VNet as our benchmark detectors.  
You can find and configure all the arguments for the program in the `tools/det/Makefile` file.

## Preparation

Choose one of our supported datasets, for example, [V2X-Sim](/datasets/v2x_sim), and perform its data preparation steps based on our documentation.


## Training

Train benchmark detectors:

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

Evaluate benchmark detectors:

- Lowerbound
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
make test_w infernce=activated

# When2com with no cross road (RSU) data
make test_w_nc infernce=activated
```

- When2com_warp
```bash
# When2com_warp
make test_warp infernce=activated

# When2com_warp with no cross road (RSU) data
make test_warp_nc infernce=activated
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
make test_warp inference=argmax_test
```


## Results
|  **Method**   | **AP@0.5 w/o RSU** | AP@0.5 w/ RSU | **Δ** | AP@0.7 w/o RSU | **AP@0.7 w/ RSU** |   Δ   |
| :-----------: | ------------------ | ------------- | ----- | -------------- | ----------------- | :---: |
|  Lower-bound  | 61.67              | 66.17         | +4.5  | 57.34          | 59.60             | +2.26 |
|   When2com    | 59.44              | 66.83         | +7.39 | 55.21          | 61.15             | +5.94 |
| When2com_warp | 61.49              | 63.74         | +2.25 | 55.85          | 58.59             | +2.74 |
|    Who2com    | 59.44              | 68.83         | +7.39 | 55.22          | 61.49             | +6.27 |
| Who2com_warp  | 61.49              | 63.74         | +2.25 | 55.85          | 58.59             | +2.74 |
|    V2VNet     | 78.89              | 82.56         | +3.67 | 74.22          | 78.61             | +4.39 |
|   DiscoNet    | 79.43              | 83.31         | +3.88 | 74.57          | 79.38             | +4.81 |
|  Upper-bound  | 81.30              | 87.38         | +6.08 | 78.09          | 83.33             | +5.24 |


