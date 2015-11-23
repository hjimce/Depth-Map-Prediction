=========================================================================
Depth Map Prediction from a Single Image using a Multi-Scale Deep Network
=========================================================================

Authors: David Eigen, Christian Puhrsch and Rob Fergus

Email:   deigen@cs.nyu.edu, cpuhrsch@nyu.edu, fergus@cs.nyu.edu


Requirements
-------------

* theano
* numpy, scipy
* PIL or Pillow


Running the Demo
-----------------

The demo loads the depth prediction network, compiles a theano function for
inference, and infers depth for a single image.  To run:

> THEANO_FLAGS=device=gpu0 python demo_depth.py

This should create a file called "demo_nyud_depth_prediction.png" with the
predicted depth for the input "demo_nyud_rgb.jpg".  (Substitute the gpu you
want to run on for gpu0).



Other Information
------------------

This tree contains code for depth prediction network inference.  While there is
some code relating to training, much of the training code including most data
processing is not provided here.  We may release this in the future, however.

While developing this project, we made a few modifications in theano not
currently part of the main codeline.  While the above instructions should work
for inference on a current unmodified theano build, it may take up more GPU
memory than needed due to use of test values for shape information.  The git
patch file "theano_test_value_size.patch" is also included and might be used to
enable this feature on your own tree.
