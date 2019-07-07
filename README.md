
# Feature Denoising for Improving Adversarial Robustness

Code and models for the paper [Feature Denoising for Improving Adversarial Robustness](https://arxiv.org/abs/1812.03411), CVPR2019.

## Introduction

<div align="center">
  <img src="teaser.jpg" width="700px" />
</div>

By combining large-scale adversarial training and feature-denoising layers,
we developed ImageNet classifiers with strong adversarial robustness.

Trained on __128 GPUs__, our ImageNet classifier has 42.6% accuracy against an extremely strong
__2000-steps white-box__ PGD targeted attack.
This is a scenario where no previous models have achieved more than 1% accuracy.

On black-box adversarial defense, our method won the __champion of defense track__ in the
[CAAD (Competition of Adversarial Attacks and Defenses) 2018](http://hof.geekpwn.org/caad/en/index.html).
It also greatly outperforms the [CAAD 2017](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack) defense track winner when evaluated
against CAAD 2017 black-box attackers.

This repo contains:

1. Our trained models, together with the evaluation script to verify their robustness.
   We welcome attackers to attack our released models and defenders to compare with our released models.

2. Our distributed adversarial training code on ImageNet.

Please see [INSTRUCTIONS.md](INSTRUCTIONS.md) for the usage.

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Citation

If you use our code, models or wish to refer to our results, please use the following BibTex entry:
```
@InProceedings{Xie_2019_CVPR,
  author = {Xie, Cihang and Wu, Yuxin and van der Maaten, Laurens and Yuille, Alan L. and He, Kaiming},
  title = {Feature Denoising for Improving Adversarial Robustness},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```
