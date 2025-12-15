***flatcad***
---
***flatcad*** is a framework designed for comparing various methods used for neural signed distance function training.

This repository contains:
* A pytorch implementation of a comparison framework, comparing state-of-the-art Neural-SDF reconstruction methods, like: [_DiGS_](https://chumbyte.github.io/DiGS-Site/), [_NeuralSingularHessian_](https://www.bearprin.com/publications/neural-singular-hessian23wang/) and [_NeurCADRecon_](https://qiujiedong.github.io/publications/NeurCADRecon/).
* An official implementation of [_FlatCAD_](https://flatcad.github.io/), as described in [_FlatCAD: Fast Curvature Regularization of Neural SDFs for CAD Models_](https://arxiv.org/abs/2506.16627).
* A official implementation of finite difference losses, as described in [_A Finite Difference Approximation of Second Order Regularization of Neural-SDFs_](https://arxiv.org/abs/2511.08980), 


## 🔧 Setup
The repository uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) dependency manager for environment setup and package installation, and relies on [`just`](https://github.com/casey/just) for shortcut command execution. Please follow the tool specific installation guides.

To install all dependencies please run:
```
uv venv
source .venv/bin/activate
uv sync
```
To examine the setup please look at `pyproject.toml`

## 🚀 Getting Started
Start experimenting by running training for a selected method using the just command:

```sh
just run [method]
```
Available Methods: `digs`, `nsh`, `nsh-fd`, `ncr`, `ncr-fd`, `odw` or `odw-fd`.

Alternatively, you can run the training script manually:
```
python src/train.py --config ./configs/train_odw.yaml
```

using a `.yaml` config file from the `./configs` directory:

## 📚 Citation
If you find this work useful in your research, please cite:
```bibtex
@article{Yin_2025,
  title={FlatCAD: Fast Curvature Regularization of Neural SDFs for CAD Models},
  volume={44},
  ISSN={1467-8659},
  url={http://dx.doi.org/10.1111/cgf.70249},
  DOI={10.1111/cgf.70249},
  number={7},
  journal={Computer Graphics Forum},
  publisher={Wiley},
  author={Yin, Haotian and Plocharski, Aleksander and Wlodarczyk, Michal Jan and Kida, Mikolaj and Musialski, Przemyslaw},
  year={2025},
  month=oct 
}

@article{yin2025finite,
  title={A Finite Difference Approximation of Second Order Regularization of Neural-SDFs},
  author={Yin, Haotian and Plocharski, Aleksander and Wlodarczyk, Michal Jan and Musialski, Przemyslaw},
  journal={arXiv preprint arXiv:2511.08980},
  year={2025}
}

@misc{yin2025schedulingoffdiagonalweingartenloss,
  title={Scheduling the Off-Diagonal Weingarten Loss of Neural SDFs for CAD Models}, 
  author={Haotian Yin and Przemyslaw Musialski},
  year={2025},
  eprint={2511.03147},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2511.03147}, 
}

@misc{fan2025jointneuralsdfreconstruction,
  title={Joint Neural SDF Reconstruction and Semantic Segmentation for CAD Models}, 
  author={Shen Fan and Przemyslaw Musialski},
  year={2025},
  eprint={2510.03837},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2510.03837}, 
}
```
## 🙏 Acknowledgement

* Credits to [DiGS](https://github.com/Chumbyte/DiGS) authors for sharing their method, which has been adapted for this framework:
  ```bibtex
  @inproceedings{ben2022digs,
  title={DiGS: Divergence guided shape implicit neural representation for unoriented point clouds},
  author={Ben-Shabat, Yizhak and Hewa Koneputugodage, Chamin and Gould, Stephen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19323--19332},
  year={2022}
  }
  ```
* Credits to [NeuralSingularHessian](https://github.com/bearprin/Neural-Singular-Hessian) authors for sharing their method, which has been adapted for this framework:
  ```bibtex
  @article{zixiong23neuralsingular,
  author = {Zixiong Wang, Yunxiao Zhang, Rui Xu, Fan Zhang, Pengshuai Wang, Shuangmin Chen, Shiqing Xin, Wenping Wang, Changhe Tu},
  title = {Neural-Singular-Hessian: Implicit Neural Representation of Unoriented Point Clouds by Enforcing Singular Hessian},
  year = {2023},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {42},
  number = {6},
  doi = {10.1145/3618311},
  publisher = {ACM}
  }
  ```

* Credits to [NeurCADRecon](https://github.com/QiujieDong/NeurCADRecon/) authors for sharing their method, which has been adapted for this framework:
  ```bibtex
  @article{Dong2024NeurCADRecon,
  author={Dong, Qiujie and Xu, Rui and Wang, Pengfei and Chen, Shuangmin and Xin, Shiqing and Jia, Xiaohong and Wang, Wenping and Tu, Changhe},
  title={NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature},
  journal={ACM Transactions on Graphics},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  year={2024},
  month={July},
  volume = {43},
  number={4},
  doi={10.1145/3658171},
  keywords = {CAD model, unoriented point cloud, surface reconstruction, signed distance function, Gaussian curvature}
  }
  ```

* Thanks to the 
[SIREN codebase](https://github.com/vsitzmann/siren) and the [IGR](https://github.com/amosgropp/IGR)/[SAL](https://github.com/amosgropp/IGR) codebases off whom original repos built upon. 


## ⚖️ License
See [LICENSE](LICENSE) file.



