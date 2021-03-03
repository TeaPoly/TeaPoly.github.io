---
layout: post
title:  "如何手动支持混合精度训练"
subtitle: "TensorFlow 1.0 版本支持混合精度训练"
date:   2021-03-02 12:00:45
categories: [tech]
---

最近阅读了Facebook 的[BENCHMARKING LF-MMI, CTC AND RNN-T CRITERIA FOR STREAMING ASR](https://arxiv.org/pdf/2011.04785.pdf) paper 之后，看到了这么一句话，让我很是心动。

> mixed-precision training was implemented in which 16-bit float numbers (fp16) are used instead of 32-bit ones (fp32), which leads to another ≃ 50% memory usage reduction (another 2x gain on batch size), with some loss on precision but compensated later by larger batch sizes. 

我们训练模型一般都是用的是单精度 (FP32) 浮点表示 ，但如果我们使用半精度(FP16)的浮点训练，可有效地降低显存开销，但是一般情况下，模型精度的下降可能会伴随准确率的下降，好在 FP16 可显著降低显存占用，因此可支持训练更大 batch size，最终效果可能会有更好的效果，另外最主要的是可有效地提升训练效率，减少推理时的开销。

下面着重介绍在 TensorFlow 1.0 版本下，如何手动支持混合精度训练，这部分内容主要参考了 [搞定大模型训练](https://xueyouluo.github.io/how-to-train-big-models/)，更多关于混合精度训练要点可以参考原文。

1. 在卷积或矩阵乘等耗时计算的地方，将输入修改成 FP16 的数据类型。

    ```python
    if dtype == tf.float16:
        inputs = tf.cast(inputs, dtype=dtype)
        kernel = tf.cast(kernel, dtype=dtype)
    return tf.matmul(inputs, kernel)
    ```

2. 可训练的参数都是 FP32，只有在前向和后向传播的时候转换为 FP16。详细可参考 [BERT](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/gpu_environment.py) 代码。

    ```python
    def float32_variable_storage_getter(getter, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, *args, **kwargs):
       """Custom variable getter that forces trainable variables to be stored in
          float32 precision and then casts them to the training precision.
       """
       storage_dtype = tf.float32 if trainable else dtype
       variable = getter(name, shape, dtype=storage_dtype,
                         initializer=initializer, regularizer=regularizer,
                         trainable=trainable,
                         *args, **kwargs)
       if trainable and dtype != tf.float32:
           variable = tf.cast(variable, dtype)
       return variable

    def get_custom_getter(compute_type):
       return float32_variable_storage_getter if compute_type == tf.float16 else None

    with tf.variable_scope("Logits", reuse=tf.AUTO_REUSE, custom_getter=get_custom_getter(tf.float16)):
       return logit(encoder)
    ```

3. Softmax 计算的采用 FP32，不然会出现 NAN 或者调用 loss 函数直接报错的问题，这里以 ctc loss 举例，我们都会直接给 loss 函数传输 logits，所以需要在 FP16 的 logits 传入之前转换成 FP32.

    ```python
    if logits.dtype == tf.float16:
      logits = tf.cast(logits, dtype=tf.float32)
    ctc_loss = tf.nn.ctc_loss(
      dense_to_sparse(labels, label_length),
      inputs=logits,
      sequence_length=logit_length,
      ignore_longer_outputs_than_inputs=True,
      time_major=True,
      preprocess_collapse_repeated=False
    )
    ```

4. 我们还需要做 loss scale，一般我们没法提前确定到底要设置多大的 scale，比较好的方法是在训练的时候动态调整 scale，可使用 `LossScaleManager` 和 `LossScaleOptimizer` 两个类，其中 `LossScaleOptimizer `可以看作是在 `tf.train.AdamOptimizer`基础上再封装了一层，因此如果你想继续拿到 Optimizer 之前的属性，比如学习率，就可以这样调用 `optimizer._opt._lr_t`：

    ```python
    # Create a basic optimizer
    optimizer = WarmUpAdam(global_step, optimizer_config)

    # Choose a loss scale manager which decides how to pick the right loss scale
    # throughout the training process.
    loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)

    # Wraps the original optimizer in a LossScaleOptimizer.
    optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
    ```

5. 这里还有一个额外需要补充的 trick，因为 batch norm 等基于统计的向量是基于 FP16 的，所以如果我们此前训练个一个 FP32 的模型，想直接 restore 包括统计值在内的所有变量就需要做一个额外的转换工作。参考 [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v2/sample/tensorflow_bert/ckpt_type_convert.py)

    ```python
    class CastFromFloat32SaverBuilder(BaseSaverBuilder):
       # Based on tensorflow.python.training.saver.BulkSaverBuilder.bulk_restore
       def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                        restore_sequentially):
           restore_specs = []
           for saveable in saveables:
               for spec in saveable.specs:
                   restore_specs.append((spec.name, spec.slice_spec, spec.dtype))
           names, slices, dtypes = zip(*restore_specs)
           restore_dtypes = [tf.float32 if dtype.base_dtype==tf.float16 else dtype for dtype in dtypes]
           # print info
           for i in range(len(restore_specs)):
               print(names[i], 'from', restore_dtypes[i], 'to', dtypes[i].base_dtype)
           with tf.device("cpu:0"):
               restored = io_ops.restore_v2(
                   filename_tensor, names, slices, restore_dtypes)
               return [tf.cast(r, dt.base_dtype) for r, dt in zip(restored, dtypes)]

    saver = tf.train.Saver(builder=CastFromFloat32SaverBuilder())
    ```

以上几个要点就是我实践下来可用的方案，我的一个小的模型从 GPU 5Gb 的占用减少到 2.7 Gb。准确率未见明显的衰减。
