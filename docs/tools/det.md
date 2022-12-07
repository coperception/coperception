# Detection

We implement lowerbound, upperbound, when2com, who2com, V2VNet as our benchmark detectors. Please see more details in our paper.

## Preparation

- Download V2XSIM datasets from our [website](https://ai4ce.github.io/V2X-Sim/index.html)
- Run the code below to generate preprocessed data
```bash
make create_data
```
You might want to consult `./Makefile` for all the arguments you can pass in.  
For example, the target for `create_data` is:
```bash
create_data:
	python create_data_det.py \
	--root $(original_data_path) \
	--split $(split) \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end) \
	--savepath $(create_data_save_path) \
	--from_agent $(from_agent) \
	--to_agent $(to_agent)
```
You should at least set `original_data_path` to the path of V2X-Sim dataset on your machine, and `create_data_save_path` to the path of the folder where you want to save the preprocessed data.  
You can set the variables at the top of `Makefile`, or you can pass them in as arguments.  
For other arguments, please see the comments in `Makefile`.


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

- When2com with pose information
```bash
# with RSU
make test_warp inference=activated

# no RSU
make test_warp_no_rsu inference=activated
```

- Who2com with pose information
```bash
# with RSU
make test_warp inference=argmax_test

# no RSU
make test_warp_no_rsu inference=argmax_test
```


## Results

Column `Δ` indicates the performance gain or loss when RSU is involved during training.  

|  **Method**   | **AP@0.5 w/o RSU** | AP@0.5 w/ RSU | **Δ** | AP@0.7 w/o RSU | **AP@0.7 w/ RSU** |   Δ   |
| :-----------: | :----------------: | :-----------: | :---: | :------------: | :---------------: | :---: |
|  Lower-bound  | 49.90              | 46.96         | -2.94  | 44.21          | 42.33             | -1.88 |
|  Co-lower-bound  | 43.99              | 42.98         | -1.01  | 39.10          | 38.26             | -0.84 |
|   When2com    | 44.02              | 46.39         | +2.37 | 39.89          | 40.32             | +0.43 |
| When2com* | 45.35              | 48.28         | +2.93 | 40.45          | 41.43             | +0.68 |
|    Who2com    | 44.02              | 46.39         | +2.37 | 39.89          | 40.32             | +0.43 |
| Who2com*  | 45.35              | 48.28         | +2.93 | 40.45          | 41.13             | +0.68 |
|    V2VNet     | 68.35              | 72.08         | +3.73 | 62.83          | 65.85             | +3.02 |
|   DiscoNet    | 69.03              | 72.87         | +3.84 | 63.44          | 66.40             | +2.96 |
|  Upper-bound  | 70.43              | 77.08         | +6.65 | 67.04          | 72.57             | +5.53 |

