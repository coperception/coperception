<div align="center">   
  
# CoPerception
  <p align="center">
    SDK for collaborative perception.
  </p>

[![Documentation Status](https://readthedocs.org/projects/coperception/badge/?version=latest)](https://coperception.readthedocs.io/en/latest/?badge=latest)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![GitLab issues total](https://badgen.net/github/issues/coperception/coperception)](https://gitlab.com/ai4ce/V2X-Sim/issues)
[![GitHub stars](https://img.shields.io/github/stars/coperception/coperception.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/ai4ce/V2X-Sim/stargazers/)
---

<img src="https://raw.githubusercontent.com/yifanlu0227/v2xsim_vistool/master/gifs/scene_overview_Mixed.gif" width="600px"/>

<img src="https://ai4ce.github.io/V2X-Sim/img/scene_72.gif" width="1000px"/>

</div>

## Install:
1. Clone this repository.
2. `cd` into the cloned repository.
3. Install `coperception` package with pip:
    ```bash
    pip install -e .
    ```

## Docs:
Please refer to our docs website for detailed documentations:
https://coperception.readthedocs.io/en/latest/ 

## Supported models:
- [DiscoNet](https://arxiv.org/abs/2111.00643)
- [V2VNet](https://arxiv.org/abs/2008.07519)
- [When2com](https://arxiv.org/abs/2006.00176)
- [Who2com](https://arxiv.org/abs/2003.09575)

Download checkpoints: [Google Drive (US)](https://drive.google.com/drive/folders/1NMag-yZSflhNw4y22i8CHTX5l8KDXnNd)  
See `./tools/det` and `./tools/seg` for model performance

## Supported datasets:

- [x] [V2X-Sim](https://ai4ce.github.io/V2X-Sim/)
- [ ] [DAIR-V2X](https://thudair.baai.ac.cn/index) (coming soon)

## Related works:
- [DiscoNet Github repo](https://github.com/ai4ce/DiscoNet)
- [V2X-Sim Github repo](https://github.com/ai4ce/V2X-Sim)

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