# Simple Framework for Contrastive Learning

[![Paper](http://img.shields.io/badge/Paper-arXiv.2002.05709-B3181B?logo=arXiv)](https://arxiv.org/abs/2002.05709)
[![Paper](http://img.shields.io/badge/Paper-arXiv.2006.10029-B3181B?logo=arXiv)](https://arxiv.org/abs/2006.10029)

<div align="center">
  <img width="50%" alt="SimCLR Illustration" src="https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif">
</div>
<div align="center">
  An illustration of SimCLR (from <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html">our blog here</a>).
</div>

```
python3 -m official.vision.beta.projects.simclr.train \
  --mode=train_and_eval \
  --experiment=simclr_pretraining_imagenet \
  --model_dir={MODEL_DIR} \
  --config_file={CONFIG_FILE}
```

## Cite

[SimCLR paper](https://arxiv.org/abs/2002.05709):

```
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```

[SimCLRv2 paper](https://arxiv.org/abs/2006.10029):

```
@article{chen2020big,
  title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
  author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2006.10029},
  year={2020}
}
```

## Disclaimer
This is not an official Google product.