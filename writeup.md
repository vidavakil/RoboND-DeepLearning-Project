## Project: Follow Me
---

# Steps to complete the project:

- Clone the project repo [here](https://github.com/udacity/RoboND-DeepLearning-Project.git)
- Fill out the TODO's in the project code as mentioned [here](https://classroom.udacity.com/nanodegrees/nd209/parts/c199593e-1e9a-4830-8e29-2c86f70f489e/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/b0044631-d356-4ff4-9fd9-4102c28a2efa?contentVersion=1.0.0&contentLocale=en-us)
- Optimize your network and hyper-parameters.
- Train your network and achieve an accuracy of 40% (0.40) using the Intersection over Union IoU metric which is final_grade_score at the bottom of your notebook.
- Make a brief writeup report summarizing why you made the choices you did in building the network.

[//]: # (Image References)

[image0]: ./misc_images/follow_me_snapshot.png
[image1]: ./misc_images/network_architecture.png
[image2]: ./misc_images/follow_me_training_loss.png

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner. The document can be submitted either in either Markdown or a PDF format.

You're reading it!

### Project Description

The goal of this project is to build a neural network model for image segmentation, and to train, validate, and deploy it in the Follow Me project, to enables a drone to track and follow a single hero target using the model.

![alt text][image0] 

### Network Architecture

Following the course instructions, the network architecture used for this project is an FCN (Fully Convolutional Network), where all layers are convolutional. The network consists of an encoder and a decoder, separated by a 1x1 convolutional layer, with skip connections between the encoder and decoder layers. 

- The encoder consists of 3 separable convolution layers (encoder_blocks). 
- The 1x1 convolution layer in the FCN, is a regular convolution. 
- The decoder consists of 3 bilinear upsampling layers (decoder_blocks).

The encoder and the 1x1 convolution layers use Relu activation function and batch normalization. 

Below image shows a block diagram of the network.

![alt text][image1] 

Below are some reasons why FCNs, and this particular architecture, is used for the image segmentation task at hand.

- In convolutional networks for image classification, the convolutional layers are followed by a number of fully connected layers. The output of the last convolutional layer is flattened and input to the first fully connected layer. Since the input size of the fully connected layer is fixed, the size of the images that can be fed to such classification network is also fixed. In contrast, FCNs do not have fully connected layers, and all of their layers are convolutional. As a result, they can handle images of arbitrary sizes. 

- In convolutional networks for image classification the output of the fully convolutional layers is fed to a softmax, the output of which is used to determine the class of object identified in the image. In contrast, the output of an FCN is an image itself, where each pixel of the image is labeled with the object class it belongs to. In the follow me project, there are 3 output classes, one for the hero (marked with blue), the other people (marked with green), and the background (marked with red).

- The encoder part of the FCN consists of three layers of depth-wise separable convolutional layers. Separable convolution layers comprise of a convolution performed over each channel of the input, followed by a 1x1 convolution that combines the results into the output channels. The number of parameters of a separable convolutional layer (as shown in an example in the lecture notes) is much less than the number of parameters in a regular convolutional layer with the same number of input and output channels. The reduced number of parameters make separable convolutions more efficient and performant for inference, and help reduce the chance of overfitting during training.

- Batch normalization is a technique that is used to normalize the distribution of the inputs to a layer over the given input batch. Normalized (versus skewed) batches make the task of learning and gradient descent easier, because the loss surface as a function of network weights will be locally more like a nice round bowl, than a stretched and skewed one. On a nice bowl, gradients always point to the bottom of the bowl, thus reaching the bottom of the bowl happens faster.

- The purpose of the 1x1 convolution in the middle of the FCN network is to condense the input image into a smaller image, with many more channels, that capture important features about the image. A 1x1 convolution takes each pixel of an N-channel input tensor, and linearly combines them into the corresponding pixel of an M-channel output tensor, using shared weights across all pixel locations. 

- The decoder part of the network then takes the condensed image and its channels and gradually upsamples it into the final segmented output image. In doing that, the decoder layers take input also from the encoder layers, via skip connections that pass along higher resolution local information to the decoder layers. 

- The decoder layers that are used in this project use bilinear upsampling layers (an alternative would have been transposed convolution layers). As describe in the course, these convolutional layers upsample their input image (and its channels) via a process of iterative interpolations, filling the output pixels by interpolating between the known input and/or computed output pixels. The output of the bilinear upsampling layer is concatenated with the output of an encoder_block, before being fed into a separable convolutional network with a stride ```(1, 1)```.

The code blocks below show my implementation of the ```encoder_block()``` and the ```decoder_block()```:

```
def encoder_block(input_layer, filters, strides):

    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

Notice that by construction, all kernel sizes are fixed at 3x3 for all convolutional blocks (please refer to the implementation of ```separable_conv2d_batchnorm()```). 

```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    output_layer = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([output_layer, large_ip_layer], axis=-1)
    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output_layer, filters, (1, 1))

    return output_layer
```

I use a stride of ```(1, 1)``` at the final layer of the ```decoder_block```. This is the correct choice for a decoder_block, as otherwise the upsampled image will be downsampled again.

### Architecture Details

For the FCN model, I used 3 ```encoder_blocks```, and 3 ```decoder_blocks``` as shown below:

```
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_1 = encoder_block(inputs, 32, (2, 2))  # was 8, 32, 128
    encoder_2 = encoder_block(encoder_1, 128, (2, 2))
    encoder_3 = encoder_block(encoder_2, 256, (2, 2)) # was 512

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_1x1 = conv2d_batchnorm(encoder_3, 512, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_1 = decoder_block(conv_1x1, encoder_2, 256) # was 512
    decoder_2 = decoder_block(decoder_1, encoder_1, 64) # was 128
    decoder_3 = decoder_block(decoder_2, inputs, 3)
    
    x = decoder_3
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

The ```fcn_model``` has a final convolutional layer with ```num_classes``` outputs. For the __Follow Me__ project, ```num_classes = 3``` and thus it is the same as the number of filters of the last ```decoder_block```. Following the last ```decoder_block``` with another convolutional layer with the same size and number of filters sounds a bit redundant, and I must have arrived at this combination at some point along my multiple trials without noticing the coincidence. In a future attempt, I can try out a different number of filters for the last ```decoder_block``` to better justify its role in the network. But for now, the above FCN works well enough.

For the ```encoder_blocks```, I consistently use a stride of ```(2, 2), which achieves a downsampling effect. The dimensions of the image are shrunk by a factor of 2 going through each of the layers. Meanwhile, I increase the number of filters by a multiple of 2 from one layer of the encoder to the next, increasing the number of features that are captured. Lower layer of the encoder learn to detect simple features such as lines and edges, while higher layers learn ever more complex patterns, features, and objects. 

The maximum number of filters that I use is in the middle 1x1 convolutional layer, and it is 512. This worked for me, but I cannot say if that is a requirement. It only made intuitive sense to me that the number of filters has to increase along the encoder blocks and then decrease along the decoder blocks, all the way back to the desired number of output classes.

I use two skip connections in this network. One is an outer skip connection between the larger (in terms of image size) ```encoder_1``` and ```decoder_2``` blocks, and one inner skip connection between the smaller (in terms of image size) ```encoder_2``` and ```decoder_1``` blocks.

### Tuning Architecture Parameters

Before I reached an accuracy of 40% using the provided training set, I had used a networks that in some of the encoder/decoder layers has twice as many filters as shown above, and I had achieved only up to 36% final score. With larger networks, the validation curves were more unstable and had larger and more frequent overshoots. The networks were overfitting. Reducing the number of filters had a noticeable impact on increasing the final score.

I used the GPU capability of the Udacity notebook to train my networks, and used about 11 hours of training, over about 4 different hyper parameter settings (including the choice of the filter sizes). Unfortunately, I did not do a thorough job of book-keeping the results (e.g., keeping snapshots of train/validation loss charts, or the exact parameters I had used in each trial), and I used each trial to get a sense of which direction to move (like a gradient step).

The above network (together with other hyper parameters I used), gave me almost exactly 40% final_score. Since it was very borderline, I tried improving it by augmenting the training data, by simply creating a flipped version of each image and mask (label) file in the provided training set. I used the following code block (ran exactly once) to augment the data.

```
"""
DATA AUGMENTATION. Make sure you run this only once, ever!
"""
def augment_folder(folder, file_type):
    files = glob.glob(os.path.join(folder, '*.' + file_type))
    for f in files:
        original_image = misc.imread(f)
        flipped_image = np.fliplr(original_image)
        fname_parts = os.path.split(f)
        flipped_base = 'f_' + fname_parts[-1]
        misc.imsave(os.path.join(folder, flipped_base), flipped_image, format=file_type)
        original_base = 'o_' + fname_parts[-1]
        os.rename(f, os.path.join(folder, original_base))

images_folder=os.path.join('..', 'data', 'train', 'images')
augment_folder(images_folder, 'jpeg')
masks_folder=os.path.join('..', 'data', 'train', 'masks')
augment_folder(masks_folder, 'png')
```

### Tuning Hyper Parameters For Training

Training the model also requires choosing values for a set of hyper parameters. The following are the values that I used for training my submitted model:

```
learning_rate = 0.01
batch_size = 64
num_epochs = 25
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

I only played with the first three parameters, ```learning_rate```, ```batch_size```, and ```num_epochs```.
Other values that I tried for learning_rate were 0.05 and 0.03. The smaller learning rate helped make the loss curves smoother, and that's why I settled at 0.01.
Initially I tried a ```batch_size``` of 128, and at the time I was getting a score of only 36% (or maybe 33%) anyway, perhaps for other reasons. Since specially at the beginning the number of filters I had were almost double, and I since I was not using any extra training data, I reduced the ```batch_size``` form 128 to 64 after the very first trial, and kept it there.
As for number of epochs, initially I used a value of 10, which was too small. Since it had caused my training to stop too early, I increased it to 25. But then I noticed that with all other parameters I was trying out, 25 would often lead to overfitting before the training ended. At some point, I thought it was necessary to be able to interrupt the training at the early signs of overfitting, and yet be able to checkpoint the model. Thus I equipped the provided code with the ability to do checkpointing, as shown below (by defining a ```checkpoint``` op and adding it to the ```callbacks``` parameter passed to  ```model.fit_generator()```). 

```
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

from workspace_utils import active_session
# Keeping Your Session Active
with active_session():
    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                   data_folder=os.path.join('..', 'data', 'train'),
                                                   image_shape=image_shape,
                                                   shift_aug=True)

    val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                 data_folder=os.path.join('..', 'data', 'validation'),
                                                 image_shape=image_shape)

    logger_cb = plotting_tools.LoggerPlotter()
    run_num = 'run_5'
    checkpoint_path = os.path.join('..', 'data', 'weights', "model_weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 monitor='val_loss', 
                                                 verbose=1, 
                                                 save_best_only=True, 
                                                 save_weights_only=True, 
                                                 mode='auto', 
                                                 period=1)
    callbacks = [logger_cb, checkpoint]

    model.fit_generator(train_iter,
                        steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                        epochs = num_epochs, # the number of epochs to train for,
                        validation_data = val_iter, # validation iterator
                        validation_steps = validation_steps, # the number of batches to validate on
                        callbacks=callbacks,
                        workers = workers)
```

The code creates a checkpoint after each epoch if the validation loss is better than the last checkpoint. I did not end up using this feature, as the set of parameters that I was converging at did not necessitate a need to early stop the training, and for the submitted model I let the training continue to completion (for 25 epochs).

Below shows the training curve for the submitted model:

![alt text][image2] 

### Results

The submitted model achieves a final score of %41 using an augmented training set (with flipped images). When the model was tried with the Follow Me simulator, the drone was able to quickly locate the hero (with only a transient loss of the target), and once found, it was able to consistently and robustly follow the hero. This is not surprising, given the high accuracy (89%) of identifying the hero while following behind the hero. But given that the accuracy (and IoU) of detecting the hero is much lower from far away (%23), the drone may have a harder time first detecting the hero in a large crowd and from far away, and may need to do a longer patrol of the environment before it is able to detect the target from a close distance.

This model is of course trained for a single target/hero, that is differentiated from other people in the environment based on her attire, hair and other features. 
If this drone were to follow another person, or animal, it has to be trained from scratch with sufficient data in multiple different crowd scenarios, and distances, etc.

### Future Enhancements

The accuracy can be further improved by collecting much more data (especially for the far away scenarios) using the simulator. This is something that I did not get a chance to do. I also did not further explore the state space of hyper parameters, once I achieved the 40% mark. The main reason was time, as even GPU runs for the range of parameters I was choosing was taking about 2-2.5 hours each. Moreover, manually tuning hyper parameters of a network can be tedious, and is more like an art form, also requiring some good intuition. Using automated systems (such as Google's Vizier) that help auto-tune hyper parameters of a network would greatly help with achieving higher performance for models and projects like this.

