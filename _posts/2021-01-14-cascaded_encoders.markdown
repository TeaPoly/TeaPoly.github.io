---
layout: post
title:  "级联编码器 (Cascaded Encoders) 流式离线混合模型"
subtitle: "流式+离线语音识别"
date:   2021-01-14 15:00:45
categories: [research]
---

受到 Google 一篇  [CASCADED ENCODERS FOR UNIFYING STREAMING AND NON-STREAMING ASR](https://arxiv.org/abs/2010.14606) 论文的启发，准备采用级联编码来改善 Second Pass 的识别效果。

最近也有其他离在线混合模型的论文，[TRANSFORMER TRANSDUCER: ONE MODEL UNIFYING STREAMING AND NON-STREAMING SPEECH RECOGNITION](https://arxiv.org/abs/2010.03192)，[Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition](https://arxiv.org/abs/2012.05481)，但是就工程上比较划算的方式还是这种采用级联的模型。级联编码的编码结构如下图所示：

![截屏2021-01-14 上午11.55.04](https://tva1.sinaimg.cn/large/008eGmZEgy1gmn33x8p35j30w00rqdjo.jpg)

在 Casual 编码的基础上加入了一个 non-causal 的编码，可以理解为在线+离线混合模型。在线实时显示实时编码结果的同时，还帮助离线模型分摊了一大部分计算量。

最终的 loss 计算可以引入一个权重系数：$L=λL_s +(1−λ)L_a$。当然 Google 也给出了他们认为最好的方式，那就是在模型训练过程中，采用了随机选择路径的方式来进行前馈计算和梯度的求解。

> In practice, we found that we can decrease the step-time during training by sampling from es and ea within a mini-batch using λ as the sampling rate. Therefore, for each input utterance, we stochastically choose either the causal or the non-causal processing path at each training step. This alleviates the need to compute RNN-T loss twice for each training example at each training step. With sampling, the model converges roughly after the same number of steps as a standalone streaming model.

这样可以减少训练中每次需要计算两次 Loss，在每条路径下累加相同总步数的情况下，收敛程度应该和分别计算是差不多的。为此，我也写了一段伪代码。

``` python
class CascadedCtcMultiLayer:
   """ 
  CASCADED ENCODERS FOR UNIFYING STREAMING AND NON-STREAMING ASR
​
  Ref: https://arxiv.org/abs/2010.14606
​
  Architecture:
​
      Inputs 
        |
        |             
  Causal Encoder-------Non-Causal Encoder
        |                     |
        |_____________________|
        |
      Logits
        |
      CTC Loss
​
  """
​
   def __init__(self, is_train_condition, proto_path, vocal_size,
                trainable=True, weight_item=0.5):
​
       self.is_train_condition = is_train_condition
​
       self.weight_item = tf.cast(weight_item, dtype=tf.float32)
​
       self.proto = ProtoParserCascadedCtc(proto_path)
​
       # EncoderNetwork
       self.encoder = TransformerNetwork.CtcEncoderNetwork(
           self.proto, is_train_condition, trainable)
​
       # LogitNetwork
       self.logit = TransformerNetwork.CtcLogitNetwork(
           self.proto, is_train_condition, trainable, vocal_size)
​
       # CascadedEncoders
       self.cascaded_encoder = TransformerNetwork.CascadedCtcEncoderNetwork(
           self.proto, is_train_condition, trainable)
​
   def __call__(self, inputs, sequence_lengths):
       '''
      @return:
          logit: Logit with causal network.
          cascaded_logit: Logit with non-causal network.
          sequence_lengths: sequence lengths.
      '''
       acoustic_features, sequence_lengths = self.encoder(
           inputs, sequence_lengths)
​
       logit, cascaded_logit = tf.cond(
           self.is_train_condition,
           lambda: self.training(acoustic_features, sequence_lengths),
           lambda: self.testing(acoustic_features, sequence_lengths)
      )
​
       return logit, cascaded_logit, sequence_lengths
​
   def training(self, acoustic_features, sequence_lengths):
       '''
      For each input utterance, we stochastically choose either the causal or 
      the non-causal processing path at each training step. 
      This alleviates the need to compute RNN-T loss twice for each training 
      example at each training step. With sampling, the model converges roughly 
      after the same number of steps as a standalone streaming model.
      '''
       random_val = tf.random.uniform(
           shape=(), minval=0., maxval=1., dtype=tf.float32)
​
       def causal():
           return self.logit(acoustic_features)
​
       def nocausal():
           cascaded_acoustic_features, _ = self.cascaded_encoder(
               acoustic_features, sequence_lengths)
           return self.logit(cascaded_acoustic_features)
​
       logit = tf.cond(
           tf.math.less_equal(random_val, self.weight_item),
           causal,
           nocausal
      )
​
       return logit, logit
​
   def testing(self, acoustic_features, sequence_lengths):
       logit = self.logit(acoustic_features)
​
       cascaded_acoustic_features, _ = self.cascaded_encoder(
           acoustic_features, sequence_lengths)
​
       cascaded_logit = self.logit(cascaded_acoustic_features)
​
       return logit, cascaded_logit
```

和原文采用 RNN-T loss 不同的是，我目前只使用了 CTC Loss 的实验正在进行中。其中 Logit 部分也共享了参数。

论文中采用级联结构后的结果如下：

![截屏2021-01-14 下午12.09.45](https://tva1.sinaimg.cn/large/008eGmZEgy1gmn3iwlblhj30v8084wfh.jpg)

可以看到和单向的 RNN-T 比较 causal 部分有略微提升。Non-causal 部分在 VS 数据下，相比于纯双向 RNN-T 模型有 6% 的相对差距。考虑到级联模型在在线应用上的优势，还是一个非常值得尝试的结构。

