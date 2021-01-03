---
layout: post
title:  "序列区分训练改进流式语音识别"
subtitle: "在线语音识别改进方法"
date:   2021-01-02 14:00:45
categories: [research]
---

这篇文章主要调研的是一种常见的改进在线语音识别的方法：序列区分性训练（[Sequence Discriminative Training](https://www.inf.ed.ac.uk/teaching/courses/asr/2017-18/asr12-seq.pdf)）。相信有很多人已经在 CTC/CE 的训练上遇到了瓶颈，而一些新的框架如 RNN-T，End2End 的实现，对于工程上的改动比较大。这个时候考虑序列区分性训练是一个非常实惠的方式，基于 CE/CTC 稳定下来的模型，接入序列区分性训练，收益非常可观且工程代码不用修改，只需要后期 tune 一下参数就好。文末会给出我最近看到的一些有助于实现该功能的 Repo。

这篇文章可能文字讲解的地方会比较少，因为图片的内容信息量已经很大了，同时论文的出处我会一并放出，有兴趣的朋友可以阅读下原文。

## 实时系统的训练准则对⽐

### Frame-Level Criterion

Cross Entropy:

![img](https://pic1.zhimg.com/v2-42165361f7a74b43a208d58f040951ac_b.jpeg)

CE Loss 公式

### End-to-End Criteria

- Connectionist Temporal Classification：

![img](https://pic3.zhimg.com/v2-7570163b539d762abce627e62abe391e_b.png)

CTC Loss 公式

- RNN-T:

![img](https://pic1.zhimg.com/v2-3f43ea97a2f905db00833833dd5ed8c8_b.jpeg)

RNN-T Loss 公式

### Sequence-Level Criteria

- Maximum Mutual Information：

![img](https://pic3.zhimg.com/v2-f01e5974846431a3726824f626bd8022_b.jpeg)

MMI Loss 公式

该 loss 出了对带浅层语言声学模型做基于词序列的标准解优化之外，还对竞争路径做了区分优化。

Ref：[Sequence-discriminative training of deep neural networks](http://isl.anthropomatik.kit.edu/cmu-kit/downloads/Sequence-discriminative_training_of_deep_neural_networks.pdf)

- Minimum Phone Error or state-level Minimum Bayes Risk：

![img](https://pic3.zhimg.com/v2-12c88b452cf2024fbaaa90842db1440a_b.jpeg)

sMBR Loss 公式

基于MMI的进一步优化是，对标准路径做了相似路径优化扩充，引入了序列相似度打分，旨在缩小类内距离同时放大类外距离。

Ref：[Sequence-discriminative training of deep neural networks](http://isl.anthropomatik.kit.edu/cmu-kit/downloads/Sequence-discriminative_training_of_deep_neural_networks.pdf)

下面是谷歌在做区分训练的实时训练流程，

![img](https://pic2.zhimg.com/v2-3004fb2df4cf34694f7a4cae4c1e7069_b.png)

谷歌的 sMBR 训练流程

Ref： [Sequence Discriminative Distributed Training of Long Short-Term Memory Recurrent Neural Networks](https://research.google/pubs/pub42547/)

## 论⽂中的实验对⽐

本章实验内容主要分享中英文下各种设置对于 sMBR 准则训练的影响。序列区分性方法可以基于 CE 或 CTC 预训练稳定后接入，从下面的实验来看收益在 5～10%。

### Model Units

![img](https://pic2.zhimg.com/v2-7a2e9c16b7401c2ebdb3f4498b61e889_b.png)

中文 Model Units 以及 CE/CTC sMBR 的实验对比

Ref: [Investigation of Modeling Units for Mandarin Speech Recognition Using Dfsmn-ctc-smbr](https://ieeexplore.ieee.org/document/8683859)

### Frame Rate

![img](https://pic2.zhimg.com/v2-3d3dfa2dc8044934d9f0acb878345821_b.png)

Frame Rate 对 CTC sMBR 的影响

Ref: [Investigation of Modeling Units for Mandarin Speech Recognition Using Dfsmn-ctc-smbr](https://ieeexplore.ieee.org/document/8683859)

### SDT language model

![img](https://pic3.zhimg.com/v2-7ac3045c7a9ffa499b98fd52d8faf996_b.png)

不同的 N-GRAM 模型对于 sMBR 训练的影响

Ref: [Sequence Discriminative Distributed Training of Long Short-Term Memory Recurrent Neural Networks](https://research.google/pubs/pub42547/)

### Training Strategy

![img](https://pic1.zhimg.com/v2-698e9e29a82802ad2f15e48ba0a14904_b.png)

不同的 CE 预训练阶段切换对于 MMI/sMBR 的影响

Ref: [Sequence Discriminative Distributed Training of Long Short-Term Memory Recurrent Neural Networks](https://research.google/pubs/pub42547/)

### Noisy Dataset

![img](https://pic2.zhimg.com/v2-9956e92a304fda7cbf1280b2b28aeb9d_b.jpeg)

CE-sMBR 和 CTC-sMBR 在不同数据类型下的对比

Ref: [FLAT START TRAINING OF CD-CTC-SMBR LSTM RNN ACOUSTIC MODELS](https://research.google.com/pubs/archive/44269.pdf)

## 开源项⽬

### [CTC-CRF](https://github.com/thu-spmi/CAT)

清华⼤学开源的项⽬，[欧老师](http://oa.ee.tsinghua.edu.cn/~ouzhijian/index.htm) 和学生 Keyu An, Hongyu Xiang 共同研发并开源的一套 CTC-CRF 区分训练方法，Hongyu Xiang 实现了 WFST 的解码在 GPU 中并行运算，解码速度非常快。目前在几个标准集上都有 state-of-the-art 结果。

![img](https://pic2.zhimg.com/v2-fa85dd315651d73b9d43cf952f26eb5d_b.jpeg)

CTC-CRF 在各种标准集下的表现情况

Ref：[CAT: CRF-based ASR Toolkit](https://arxiv.org/abs/1911.08747)

团队完成了 PyTorch 的 binding。由于我主要是在 TensorFlow 上实现训练流程，所以基于此，稍微重构了些代码，完成了 Tensorflow 的 binding [TeaPoly/warp-ctc-crf](https://github.com/TeaPoly/warp-ctc-crf)。另外 TensorFlow 的训练工具也在最近上传了，地址为 [TeaPoly/cat_tensorflow](https://github.com/TeaPoly/cat_tensorflow)。

使用公司内部的远场的 2MB 左右的小模型也取得了一定的收益，具体如下：

![img](https://pic4.zhimg.com/v2-39ae3052c812844a4005e5caf285c9f3_b.png)

远场识别小模型引入 CTC-CRF 后的提升

### [EESEN](https://github.com/srvk/eesen)

[Yajie Miao](http://www.cs.cmu.edu/~ymiao) 基于 Kaldi 工具完善的⼀套 CTC 训练框架。虽然没有使用目前主流的 TensorFlow/PyTorch 框架，但放到现在任然有⼀定的参考价值，⽬前尚未⽀持区分训练。

### [PyChain](https://github.com/YiwenShaoStephen/pychain)

小⽶语音组成员开发，[Daniel Povey](https://www.danielpovey.com/) 参与，基于 LF-MMI chain model。基于 PyTorch 框架开发，提供了详细的 Pipeline 样例，同时完成了 FST 和 PyTorch 的 binding。

### [tf-code-acoustics](https://github.com/datemoon/tf-code-acoustics)

[hubo](https://github.com/datemoon) 开发，从代码看应该是 Sogou 成员，作者相当勤奋，完成了MMI/MPE/sMBR/CTC/CE 的 TensorFlow 下的训练代码。和原作者沟通下来，他基于 CE/CTC-sMBR 做了尝试，相⽐于单纯 CE/CTC 都有 5% 到 10% 的收益。系统里有 Pipeline 和样例。 虽然目前没有特别完备的开发⽂档，但是作者的开源精神和勤奋程度让人佩服。
