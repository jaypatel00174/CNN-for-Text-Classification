# Movie_CNN
Movie classification using Convolution Neural Network (CNN)  

It is slightly simplified implementation of Yoon Kim's Convolutional Neural Networks for Sentence Classification paper in Tensorflow.

Data is avaiable on http://www.cs.cornell.edu/people/pabo/movie-review-data/ or you can find it on Yoon Kim's cod on git hub https://github.com/yoonkim/CNN_sentence Data files are: rt-polarity.neg and rt-polarity.pos

We have implemented it using TenserFlow Requirements and other details are as follow

Requirements

Python 3 Tensorflow > 0.12 Numpy Training

Print parameters:

./train.py --help <br />

optional arguments:<br /> 
-h, --help show this help message and exit <br />
--embedding_dim EMBEDDING_DIM Dimensionality of character embedding (default: 128) <br />
--filter_sizes FILTER_SIZES Comma-separated filter sizes (default: '3,4,5') <br />
--num_filters NUM_FILTERS Number of filters per filter size (default: 128) <br />
--l2_reg_lambda L2_REG_LAMBDA L2 regularizaion lambda (default: 0.0) <br />
--dropout_keep_prob DROPOUT_KEEP_PROB Dropout keep probability (default: 0.5) <br />
--batch_size BATCH_SIZE Batch Size (default: 64) <br />
--num_epochs NUM_EPOCHS Number of training epochs (default: 100) <br />
--evaluate_every EVALUATE_EVERY Evaluate model on dev set after this many steps (default: 100) <br />
--checkpoint_every CHECKPOINT_EVERY Save model after this many steps (default: 100) <br />
--allow_soft_placement ALLOW_SOFT_PLACEMENT Allow device soft device placement <br />
--noallow_soft_placement <br />
--log_device_placement LOG_DEVICE_PLACEMENT Log placement of ops on devices <br />
--nolog_device_placement<br />

Train:

./train.py Evaluating

./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/" Replace the checkpoint dir with the output from the training. To use your own data, change the eval.py script to load your data.

References

Convolutional Neural Networks for Sentence Classification

A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification
