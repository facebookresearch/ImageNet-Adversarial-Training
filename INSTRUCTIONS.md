
## Dependencies:

+ TensorFlow ≥ 1.6 with GPU support
+ OpenCV ≥ 3
+ Tensorpack = 0.9.1
+ horovod ≥ 0.15 with NCCL support
  + horovod has many [installation options](https://github.com/uber/horovod/blob/master/docs/gpus.md) to optimize its multi-machine/multi-GPU performance.
    You might want to follow them.
+ TensorFlow [zmq_ops](https://github.com/tensorpack/zmq_ops) (needed only for training)
+ ImageNet data in its [standard directory structure](https://tensorpack.readthedocs.io/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.ILSVRC12)


## Model Zoo:
<table>
<thead>
<tr>
<th align="left" rowspan=2>Model (expand for flags)</th>
<th align="center">error rate</th>
<th align="center" colspan=3>error rate / attack success rate</th>
</tr>
<tr>
<th align="center">clean images</th>
<th align="center">10-step PGD</th>
<th align="center">100-step PGD</th>
<th align="center">1000-step PGD</th>
</tr>
</thead>


<tbody>
<tr>
<td align="left"><details><summary>ResNet152 Baseline <a href="R152"> :arrow_down: </a> </summary> <code>--arch ResNet -d 152</code></details></td>
<td align="center">37.7%</td>
<td align="center">47.5%/5.5%</td>
<td align="center">58.4%/31.0%</td>
<td align="center">60.7%/36.1%</td>
</tr>

<tr>
<td align="left"><details><summary>ResNet152 Denoise  <a href="R152Denoise"> :arrow_down: </a> </summary> <code>--arch ResNetDenoise -d 152</code></details></td>
<td align="center">34.7%</td>
<td align="center">44.1%/4.9%</td>
<td align="center">54.6%/26.6%</td>
<td align="center">56.9%/32.7%</td>
</tr>

<tr>
<td align="left"><details><summary>ResNeXt101 DenoiseAll  <a href="X101DenoiseAll"> :arrow_down: </a> </summary><code>--arch ResNeXtDenoiseAll</code> <br> <code>-d 101</code> </details></td>
<td align="center">31.6%</td>
<td align="center">44.0%/4.9%</td>
<td align="center">55.6%/31.5%</td>
<td align="center">59.6%/38.1%</td>
</tr>
</tbody>
</table>



Note:

1. As mentioned in the paper, our attack scenario is:

   1. targeted attack with random uniform target label
   2. maximum perturbation per pixel is 16.

   We do not perform untargeted attack, nor do we let the attacker choose the target label,
   because we believe such tasks are not realistic on the 1000 ImageNet classes.

2. For each (attacker, model) pair, we provide both the __error rate__ of our model,
   and the __attack success rate__ of the attacker, on ImageNet validation set.
   A targeted attack is considered successful if the image is classified to the target label.

   If you develop a new robust model, please compare its error rate with our models.
   Attack success rate is not a reasonable metric, because then the model can cheat by making random predictions.

   If you develop a new attack method against our models,
   please compare its attack success rate with PGD.
   Error rate is not a reasonable metric, because then the method can cheat by becoming
   untargeted attacks.

3. `ResNeXt101 DenoiseAll` is the submission that won the champion of
   black-box defense track in [Competition on Adversarial Attacks and Defenses 2018](https://en.caad.geekpwn.org/).


## Evaluate White-Box Robustness:

To evaluate on one GPU, run this command:
```
python main.py --eval --load /path/to/model_checkpoint --data /path/to/imagenet \
  --attack-iter [INTEGER] --attack-epsilon 16.0 [--architecture-flags]
```

To reproduce our evaluation results,
take "architecture flags" from the first column, and set the attack iteration.
Iteration can be set to 0 to evaluate its clean image error rate.
Note that the evaluation result may have a ±0.3 fluctuation due to the
randomly-chosen target attack label and attack initialization.

Using a K-step attacker makes the evaluation K-times slower.
To speed up evaluation, run it under MPI with multi-GPU or multiple machines, e.g.:

```
mpirun -np 8 python main.py --eval --load /path/to/model_checkpoint --data /path/to/imagenet \
  --attack-iter [INTEGER] --attack-epsilon 16.0 [--architecture-flags]
```

Evaluating the `Res152 Denoise` model against 100-step PGD attackers takes about 1 hour with 16 V100s.


## Evaluate Black-Box Robustness:

We provide a command line option to produce predictions for an image directory:
```
python main.py --eval-directory /path/to/image/directory --load /path/to/model_checkpoint \
  [--architecture-flags]
```

This will produce a file "predictions.txt" which contains the filename and
predicted label for each image found in the directory.
You can use this to evaluate its black-box robustness.

## Train:

Adversarial training takes a long time and we recommend doing it only when you have a lot of GPUs.
Large models may require GPUs with 32GB memory to train.
You can use our code for standard ImageNet training as well (with `--attack-iter 0`).

To train, first start one data serving process __on each machine__:
```
$ ./third_party/serve-data.py --data /path/to/imagenet/ --batch 32
```

Then, launch a distributed job with MPI. You may need to consult your cluster
administrator for the MPI command line arguments you should use.
On a cluster with InfiniBand, it may look like this:

```
 mpirun -np 16 -H host1:8,host2:8 --output-filename train.log \
    -bind-to none -map-by slot -mca pml ob1 \
    -x NCCL_IB_CUDA_SUPPORT=1 -x NCCL_IB_DISABLE=0 -x NCCL_DEBUG=INFO \
    -x PATH -x PYTHONPATH -x LD_LIBRARY_PATH \
    python main.py --data /path/to/imagenet \
        --batch 32 --attack-iter [INTEGER] --attack-epsilon 16.0 [--architecture-flags]
```

If your cluster is managed by slurm , we provide some sample [slurm job scripts](slurm/)
for your reference.

The training code will also perform distributed evaluation of white-box robustness.

### Speed:

With 30 attack iterations during training,
the `Res152 Baseline` model takes about 52 hours to finish training on 128 V100s.

Under the same setting, the `Res152 Denoise` model takes about 90 hours on 128 V100s.
Note that the model actually does not add much computation to the baseline,
but it lacks efficient GPU implementation for the softmax version of non-local operation.
