<div align="center">   
  
## CoPerception
 <p align="center">
    SDK for collaborative perception.
</p>
</div>

---

## Install:
1. Clone this repository.
2. `cd` into the cloned repository.
3. Install `coperception` package with pip:
    ```bash
    pip install -e .
    ```

## Docs:
[![Documentation Status](https://readthedocs.org/projects/coperception/badge/?version=latest)](https://coperception.readthedocs.io/en/latest/?badge=latest)  
Please refer to our docs website for detailed documentations:
https://coperception.readthedocs.io/en/latest/ 

## Supported models:
- [DiscoNet](https://arxiv.org/abs/2111.00643)
- [V2VNet](https://arxiv.org/abs/2008.07519)
- [When2com](https://arxiv.org/abs/2006.00176)
- [Who2com](https://arxiv.org/abs/2003.09575)

## Papers cited:
DisoNet:
```
@article{li2021learning,
  title={Learning distilled collaboration graph for multi-agent perception},
  author={Li, Yiming and Ren, Shunli and Wu, Pengxiang and Chen, Siheng and Feng, Chen and Zhang, Wenjun},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={29541--29552},
  year={2021}
}
```

V2VNet:
```
@inproceedings{wang2020v2vnet,
  title={V2vnet: Vehicle-to-vehicle communication for joint perception and prediction},
  author={Wang, Tsun-Hsuan and Manivasagam, Sivabalan and Liang, Ming and Yang, Bin and Zeng, Wenyuan and Urtasun, Raquel},
  booktitle={European Conference on Computer Vision},
  pages={605--621},
  year={2020},
  organization={Springer}
}
```

When2com:
```
@inproceedings{liu2020when2com,
  title={When2com: Multi-agent perception via communication graph grouping},
  author={Liu, Yen-Cheng and Tian, Junjiao and Glaser, Nathaniel and Kira, Zsolt},
  booktitle={Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition},
  pages={4106--4115},
  year={2020}
}
```

Who2com:
```
@inproceedings{liu2020who2com,
  title={Who2com: Collaborative perception via learnable handshake communication},
  author={Liu, Yen-Cheng and Tian, Junjiao and Ma, Chih-Yao and Glaser, Nathan and Kuo, Chia-Wen and Kira, Zsolt},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6876--6883},
  year={2020},
  organization={IEEE}
}
```