<div align="center">   
  
# CoPerception
  <p align="center">
    An SDK for collaborative perception
  </p>

[![Documentation Status](https://readthedocs.org/projects/coperception/badge/?version=latest)](https://coperception.readthedocs.io/en/latest/?badge=latest)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![GitHub issues total](https://badgen.net/github/issues/coperception/coperception)](https://github.com/coperception/coperception/issues)
![GitHub issues](https://img.shields.io/github/issues/coperception/coperception)
[![GitHub stars](https://img.shields.io/github/stars/coperception/coperception.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/coperception/coperception/stargazers/)
---

<img src="https://raw.githubusercontent.com/yifanlu0227/v2xsim_vistool/master/gifs/scene_overview_Mixed.gif" width="600px"/>

<img src="https://ai4ce.github.io/V2X-Sim/img/scene_72.gif" width="1000px"/>

</div>

## Getting started:
Please refer to our docs website for detailed documentations: https://coperception.readthedocs.io/en/latest/  
### Installation
- [Installation documentations](https://coperception.readthedocs.io/en/latest/getting_started/installation/)

### Download dataset
- [V2X-Sim](https://coperception.readthedocs.io/en/latest/datasets/v2x_sim/)

How to run the following tasks:
- [Detection](https://coperception.readthedocs.io/en/latest/tools/det/)
- [Segmentation](https://coperception.readthedocs.io/en/latest/tools/seg/)
- [Tracking](https://coperception.readthedocs.io/en/latest/tools/track/)

## Supported models
- [x] [DiscoNet](https://arxiv.org/abs/2111.00643)
- [x] [V2VNet](https://arxiv.org/abs/2008.07519)
- [x] [When2com](https://arxiv.org/abs/2006.00176)
- [x] [Who2com](https://arxiv.org/abs/2003.09575)
- [ ] [V2X-ViT](https://github.com/DerrickXuNu/v2x-vit) (coming soon)

Download checkpoints: [Google Drive (US)](https://drive.google.com/drive/folders/1NMag-yZSflhNw4y22i8CHTX5l8KDXnNd)  
See `README.md` in `./tools/det`, `./tools/seg`, and `./tools/track` for model performance under different tasks.

## Supported datasets

- [x] [V2X-Sim](https://ai4ce.github.io/V2X-Sim/)
- [ ] [DAIR-V2X](https://thudair.baai.ac.cn/index) (coming soon)
- [ ] [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) (coming soon)

## Related works
- [DiscoNet Github repo](https://github.com/ai4ce/DiscoNet)
- [V2X-Sim Github repo](https://github.com/ai4ce/V2X-Sim)

## Related papers
V2X-Sim dataset:
```bibtex
@article{Li_2021_RAL,
  title = {V2X-Sim: A Virtual Collaborative Perception Dataset and Benchmark for Autonomous Driving},
  author = {Li, Yiming and Ma, Dekun and An, Ziyan and Wang, Zixun and Zhong, Yiqi and Chen, Siheng and Feng, Chen},
  booktitle = {IEEE Robotics and Automation Letters},
  year = {2022}
}
```

DisoNet:
```bibtex
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
```bibtex
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
```bibtex
@inproceedings{liu2020when2com,
  title={When2com: Multi-agent perception via communication graph grouping},
  author={Liu, Yen-Cheng and Tian, Junjiao and Glaser, Nathaniel and Kira, Zsolt},
  booktitle={Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition},
  pages={4106--4115},
  year={2020}
}
```

Who2com:
```bibtex
@inproceedings{liu2020who2com,
  title={Who2com: Collaborative perception via learnable handshake communication},
  author={Liu, Yen-Cheng and Tian, Junjiao and Ma, Chih-Yao and Glaser, Nathan and Kuo, Chia-Wen and Kira, Zsolt},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6876--6883},
  year={2020},
  organization={IEEE}
}
```

OPV2V:
```bibtex
@inproceedings{xu2022opencood,
  author = {Runsheng Xu, Hao Xiang, Xin Xia, Xu Han, Jinlong Li, Jiaqi Ma},
  title = {OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication},
  booktitle = {2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2022}
}
```
