# perforated-cnn-caffe

PerforatedCNNs accelerate convolutional neural networks (CNNs) by skipping evaluation of the convolutional layers in some of the spatial positions. See the paper for more details:

Michael Figurnov, Dmitry Vetrov, Pushmeet Kohli. PerforatedCNNs: Acceleration through Elimination of Redundant Convolutions. _Under review as a conference paper at ICLR 2016_ [[arXiv](http://arxiv.org/abs/1504.08362)].

The code is based on Caffe from October 2015. The main purpose of this code is to fine-tune large models (AlexNet and VGG16). 

See also [MatConvNet](https://github.com/mfigurnov/perforated-cnn-matconvnet) implementation of PerforatedCNNs.
Differences between the two versions:

1. Caffe version does not support perforating a network. For the purposes of the paper, we performed perforation in MatConvNet code, and then imported the network using `python/import_matconvnet.py` script.
2. Caffe version performs explicit interpolation of outputs in `ConvolutionPerforatedLayer`. This makes this implementation more self-contained, compared to MatConvNet implementation with implicit interpolation, as there is no need to tweak indices of reads of the next layer. However, this means that no memory is saved for storage of intermediate outputs, and that the speedup might be lower (especially for CPU version).

# Original Caffe README

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

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
