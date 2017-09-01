# Deep-Image-Matting
This is tensorflow implementation for paper "Deep Image Matting".

Thanks to Davi Frossard, "vgg16_weights.npz" can be found in his blog:
"https://www.cs.toronto.edu/~frossard/post/vgg16/"

2017-8-25:
Now this code can be used to train, but the data is owned by company.I'll try my best to provide code and model that can do inference.Fix bugs about memory leak when training and change one of randomly crop size from 640 to 620 for boundary security issue.This can be avoid by preparing training data more carefully. Besides, it can save model and restore pre-trained model now, and can test on alphamatting set at rum time.

2017-9-1:
Validation code and tensorboard view on 'alphamatting' dataset are added. Some bugs on compositional_loss and validation code are  fixed. Missed 'fc6' layer is added now. And the decoder structure is exactly same with paper despide of replacing unpooling with deconvolution layer which means the network is more complex than before. The weight Wi of two loss is still vague, I'm trying to find best weight structure. Currently, general boundary is easy to predit. But some details or complex foregrounds like bike is still bad. 
