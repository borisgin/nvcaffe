# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu))
and community contributors.

# NVCaffe

NVIDIA Caffe ([NVIDIA Corporation &copy;2017](http://nvidia.com)) is an NVIDIA-maintained fork
of BVLC Caffe tuned for NVIDIA GPUs, particularly in multi-GPU configurations.
Here are the major features:
* **16 bit (half) floating point train and inference support**.
* **Mixed-precision support**. It allows to store and/or compute data in either 
64, 32 or 16 bit formats. Precision can be defined for every layer (forward and 
backward passes might be different too), or it can be set for the whole Net.
* **Integration with  [cuDNN](https://developer.nvidia.com/cudnn) v7**.
* **Automatic selection of the best cuDNN convolution algorithm**.
* **Integration with v2.2 of [NCCL library](https://github.com/NVIDIA/nccl)**
 for improved multi-GPU scaling.
* **Optimized GPU memory management** for data and parameters storage, I/O buffers 
and workspace for convolutional layers.
* **Parallel data parser and transformer** for improved I/O performance.
* **Parallel back propagation and gradient reduction** on multi-GPU systems.
* **Fast solvers implementation with fused CUDA kernels for weights and history update**.
* **Multi-GPU test phase** for even memory load across multiple GPUs.
* **Backward compatibility with BVLC Caffe and NVCaffe 0.15**.
* **Extended set of optimized models** (including 16 bit floating point examples).


## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

## Useful notes

Libturbojpeg library is used since 0.16.5. It has a packaging bug. Please execute the following (required for Makefile, optional for CMake):
```
sudo apt-get install libturbojpeg libturbojpeg-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.1.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so
```