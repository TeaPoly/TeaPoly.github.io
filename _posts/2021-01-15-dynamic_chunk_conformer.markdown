---
layout: post
title:  "动态 Chunk Conformer 在线+离线混合 ASR 模型"
subtitle: "Dynamic-Chunk + Two-pass Conformer ASR"
date:   2021-01-16 06:00:45
categories: [tech]
---

这次要分享的是[出门问问](https://www.chumenwenwen.com)最近分享的一篇 Paper [Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition](http://arxiv.org/abs/2012.05481)， 他们团队还一并奉上了训练代码 [WeNet](https://github.com/mobvoi/WeNet)，是基于 [ESPnet](https://github.com/espnet/espnet) 修改而来，使用过 ESPnet 的朋友，应该是得心应手了。基于滴滴的 Athena 框架（TensorFlow 2.2) 我添加了 Dynamic chunk-based attention Conformer 的支持 [Conformer-Athena](https://github.com/TeaPoly/Conformer-Athena)。


基于 Chunk 的在线 Transformer 模型我们已经看过很多了，这次出门问问在目前有着 [SOTA](https://en.wikipedia.org/wiki/State_of_the_art) 加身的 [Conformer](http://arxiv.org/abs/2005.08100) 模型的基础上，引入动态 chunk 的模型，训练一个模型就可以同时支持从 40ms 到 1s 的在线模型，以及纯离线的 Conformer 模型，并且在引入了 CTC Prefix 在线解码 + Decoder Second Pass Resore 的离线处理逻辑后，应对在线离线混合识别的场景时显得底气十足，具体框架如下：

![截屏2021-01-16 下午1.45.14](https://tva1.sinaimg.cn/large/008eGmZEgy1gmphj2ltpnj30oe0l63zy.jpg)



## 流式处理

在线的 Conformer 模型是最近的一个大热点，常见的两种基于 Transformer 的在线处理模式是基于 chunk 和 lookahead 。最近的一些 Paper  [STREAMING AUTOMATIC SPEECH RECOGNITION WITH THE TRANSFORMER MODEL](http://arxiv.org/abs/2001.02674), [A Better and Faster End-to-End Model for Streaming ASR](http://arxiv.org/abs/2011.10798), [TRANSFORMER TRANSDUCER: ONE MODEL UNIFYING STREAMING AND NON-STREAMING SPEECH RECOGNITION](http://arxiv.org/abs/2010.03192)，可以看出 Google对于动态 lookahead context 的流式 Conformer 处理方式非常着迷，这可能是基于工程上的考量。

但从识别效果来看基于 chunk 的处理方式会优于 lookahead 的方式，具体可以参考微软的最近文章 [On the Comparison of Popular End-to-End Models for Large Scale Speech Recognition](http://arxiv.org/abs/2005.14327), 对应的 Repo 在 [StreamingTransformer](https://github.com/cywang97/StreamingTransformer)，Librispeech 的 chunk 和 lookahead 结果如下：

| Model                                   | test-clean | test-other | latency | size |
| --------------------------------------- | ---------- | ---------- | ------- | ---- |
| streaming_transformer-chunk32-conv2d    | 2.8        | 7.5        | 640ms   | 78M  |
| streaming_transformer-chunk32-vgg       | 2.8        | 7.0        | 640ms   | 78M  |
| streaming_transformer-lookahead2-conv2d | 3.0        | 8.6        | 1230ms  | 78M  |
| streaming_transformer-lookahead2-vgg    | 2.8        | 7.5        | 1230ms  | 78M  |

说到流式的 Conformer 模型，我们就应该先来看看 Conformer 转换成在线处理方式的关键点。

### 自注意力 （Self-Attention）模型

第一部分自然是 Self-Attention 的注意力求解部分。离线的处理方式通常看的是整句特征，即当前时间点往左和往右都看无限时长，这种效果显然是最佳的，但在线识别显然是不可用的。

考虑在线的处理模式，这里以 chunk 为例，我们将一段特征看作是 N 个由 chunk size 长度组成的，然后分别计算  Self-Attention。在 PyTorch 和 TensorFlow 创建多个 Chunk 分别计算的方式是有技巧可言的，我们可以基于原来离线模型的基础上，创建一个时序相关的 Mask 来模拟在线生成 chunk 的方式，具体如下图所示：

![截屏2021-01-16 下午1.46.10](https://tva1.sinaimg.cn/large/008eGmZEgy1gmphk3wm1oj30nw0as3za.jpg)

[WeNet](https://github.com/mobvoi/WeNet) 的 PyTorch 的[实现代码](https://github.com/mobvoi/wenet/blob/ee43964afd8fe1c984f030124075b9d1e463b444/wenet/utils/mask.py#L33)如下：

```python
def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device
    Returns:
        torch.Tensor: mask
    Examples:
        >>> subsequent_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, 0:ending] = True
    return ret
```

基于此我写一个 TensorFlow 的实现版本，这里不得不吐槽下 TensorFlow 的设计，确实不如 PyTorch 那么 [Pythonic](https://www.computerhope.com/jargon/p/pythonic.htm)：

```python
def subsequent_chunk_mask(size, chunk_size):
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(4, 2, 1)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    x = tf.broadcast_to(
        tf.cast(tf.range(size), dtype=tf.int32), (size, size))

    chunk_current = tf.math.floordiv(x, chunk_size) * chunk_size
    chunk_ending = tf.math.minimum(chunk_current+chunk_size, size)
    chunk_ending_ = tf.transpose(chunk_ending)
    return tf.cast(tf.math.less_equal(chunk_ending, chunk_ending_), dtype=tf.float32)
```



当然这里还有一个非常需要注意的点，就是每个样本的实际音频长度不一样大，因此需要在原来关于每个音频长度的基础上再做 Mask 生成的工作。

### 卷积模块（Conv Module）

第二部分是由于 Conformer 在 Transformer 的全局视野的基础上再引入了局部特征，虽然看上去不是一个极简的设计，但目前就实验而 SOTA 无疑。而且我认为这是一个非常有意思的设计（后面会讲到）。

这部分的核心其实是 Depthwise Conv1D 的结构，如果你此前接触过 FSMN 或者 SVDF ，理解起来就很简单了。我的理解是该结构考虑的是当前帧特征和过去、未来有限长的特征关系，通过有限脉冲响应方程（或称逐通道卷积）来完成，ESPnet 中大部分的实验采用的 kernel size 为 15 的大小，即左右分别看 7 帧，当然这部分的延迟是由右看引起的，假设我们有12层 Conformer 模型，前端 Subsamping 为 40 ms 的话，关于 Conv 这部分的总延迟就是  $ 12*7*40 = 3360ms  $ ，即 3.36 秒的延时，目前针对这部分延迟基本上是简单地移除右看部分，变成 Causal Conv Module，[部分代码](https://github.com/mobvoi/wenet/blob/ee43964afd8fe1c984f030124075b9d1e463b444/wenet/transformer/convolution.py#L44) 如下:

```python
# self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
```

其实这部分也可以根据前后填充零的帧数来控制左右视野，但是那样除了要调整 chunk 大小，还要控制卷积部分的局部视野。

基于 chunk 的流式处理采用 Causal Conv Module 除了减少延迟之外，还能减少右看对于 chunk self-attention 模型的破坏作用，这个论文[Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition](http://arxiv.org/abs/2012.05481)中有着重描述：

> The total right context depends on convolution layer’s context and the stack number of conformer layers. So this structure not only brings in additional latency, but also ruin the benefits of chunk-based attention, that the latency is indepen- dent on the network structure and could be just controlled by chunk at inference time. To overcome this issue, we use casual convolution instead.

这里还有一个很有意思的点是关于 Conv Module 的局部视野包含了相对位置的信息，这对于移除 Relative/Global Position Encoding 是有很大帮助的，Google 最新的 Paper  [A Better and Faster End-to-End Model for Streaming ASR](http://arxiv.org/abs/2011.10798) 有详细阐述：

> We speedup the training and inference by removing the original relative positional encoding. Instead, we reuse the existing convolution module that aggregates information from neighboring context to implicitly provide relative positional information. This is done by simply swapping the order of the convolution module and the multi-head self-attention module.

![截屏2021-01-16 下午2.10.01](https://tva1.sinaimg.cn/large/008eGmZEgy1gmpi8n71zcj30p606kabe.jpg)

## 动态 Chunk

关于通过动态的控制 chunk 分别的比例，文中纯离线方式在每个 Epoch 中占据 50%的比例，剩下1-25 （40ms-1s）的chunk 在剩余的 50% 是等比例分布的。具体公式如下：

![截屏2021-01-16 下午6.03.37](https://tva1.sinaimg.cn/large/008eGmZEgy1gmpozoxy1dj30nc03aq39.jpg)

PyTorch 的[代码实现](https://github.com/mobvoi/wenet/blob/ee43964afd8fe1c984f030124075b9d1e463b444/wenet/utils/mask.py#L63)如下：

```python
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)  
```



这里需要注意的是 [ESPnet](https://github.com/espnet/espnet) 在进行 Aishell 的实验时基本上 50 Epoch 就停了，而 WeNet 在动态 chunk 的训练过程中使用了 80 Epochs。我认为这是合理的，因为一次训练有多种 chunk 用以训练，每种 chunk 需要看到一定量的样本，因此需要的训练步长也要比过去单单离线的 Conformer 要长。



## CTC && AED



为了满足在线实时显示部分结果的需求。使用 chunk Conformer 的 Multi-Task 的 CTC 部分做实时解码，这部分让我联想到了 Apple 最近的文章 [Hybrid Transformer/CTC Networks for Hardware Efficient Voice Triggering](http://arxiv.org/abs/2008.02323) ，在 Transformer 的 AED 基础上，为了减少计算量仅使用 CTC 部分解码效果。对比仅用 Transformer Encoder 和 BiLSTM  的 CTC 训练结果，效果明显要更优，具体的描述如下:

> The self- attention network trained with the CTC loss (red) improves over the baseline BiLSTM network (blue). Next, the self-attention network trained with both the CTC loss and the additional de- coder yields further improvements (green), notably yielding better accuracies than the MTL version of the BiLSTM base- line (dashed blue). Finally, the MTL versions of both the self- attention networks yield significant improvements over all the baselines. 

Apple  [Hybrid Transformer/CTC Networks for Hardware Efficient Voice Triggering](https://arxiv.org/abs/2008.02323) 的文中也给出了具体结果如下：

![截屏2021-01-16 下午2.11.03](https://tva1.sinaimg.cn/large/008eGmZEgy1gmpi9oeaeej30ok0h2goh.jpg)

说回 WeNet 的框架，除了在线的 CTC Prefix Beam searcher，还引入了AED 的 rescore，即出在线结果的尾部，做一个二次修正。

![截屏2021-01-16 下午1.45.14](https://tva1.sinaimg.cn/large/008eGmZEgy1gmphj2ltpnj30oe0l63zy.jpg)

将最终的识别结果拉回到接近离线 Confomer 的效果，从 dynamic chunk 到 CTC+AED，出门问问看来是准备打造一个超级简洁的框架。

## 实验结果

最后还是要贴下 Dynamic-Chunk Conformer 在 Aishell 的 [实验结果](https://github.com/mobvoi/WeNet/tree/main/examples/aishell/s1)。

Conformer (causal convolution)

- config: conf/train_unified_conformer.yaml
- beam: 10
- num of gpu: 8
- ctc weight (used for attention rescoring): 0.5

| decoding mode/chunk size | full | 16   | 8    | 4    | 1    |
| ------------------------ | ---- | ---- | ---- | ---- | ---- |
| attention decoder        | 5.27 | 5.51 | 5.67 | 5.72 | 5.88 |
| ctc greedy search        | 5.49 | 6.08 | 6.41 | 6.64 | 7.58 |
| ctc prefix beam search   | 5.49 | 6.08 | 6.41 | 6.64 | 7.58 |
| attention rescoring      | 4.90 | 5.33 | 5.52 | 5.71 | 6.23 |

CTC + Attention Rescoring 的在不同时延下的表现可以说是非常稳定。

