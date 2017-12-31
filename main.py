import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Load the model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # Load the Graph
    graph = tf.get_default_graph()
    img_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return img_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # We need to implement encoder for the FCN-8 Architecture.
    # Since we will use this for identifying road pixels its 2 classes
    
    kernel_Init = tf.random_normal_initializer(stddev=0.01)
    kernel_Regu = tf.contrib.layers.l2_regularizer(1e-3)
    
    # 1 x 1 convolution of the last VGG Layer (7)
    # argument 1 represents the kernel size 1 and hence 1 x 1 convolution
    layer7_enc_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                      padding='same',
                                      kernel_initializer=kernel_Init,
                                      kernel_regularizer=kernel_Regu)
    
    # Upsample the 1x1 encoded layer to original image size
    # transpose layer will be 4-dimensional: (batch_size, original_height, original_width, num_classes).
    layer7_upsampled = tf.layers.conv2d_transpose(layer7_enc_out, num_classes, 4,
                                                  strides=(2,2),
                                                  padding='same',
                                                  kernel_initializer=kernel_Init,
                                                  kernel_regularizer=kernel_Regu)
    
    # 1 x 1 convolution of the VGG layer 4
    layer4_enc_out = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                      padding='same',
                                      kernel_initializer=kernel_Init,
                                      kernel_regularizer=kernel_Regu)
    
    # Peform Skip connections, element wise addition
    layer_skip_1 = tf.add(layer7_upsampled, layer4_enc_out)
    
    # Upsamping of skip connection layer output
    layer_skip_1_upsampled = tf.layers.conv2d_transpose(layer_skip_1, num_classes, 4,
                                                        strides=(2,2),
                                                        padding='same',
                                                        kernel_initializer=kernel_Init,
                                                        kernel_regularizer=kernel_Regu)
    
    # 1 x 1 convolution of VGG layer 3
    layer3_enc_out = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                      padding='same',
                                      kernel_initializer=kernel_Init,
                                      kernel_regularizer=kernel_Regu)
    
    # Peform Skip connections, element wise addition
    layer_skip_2 = tf.add(layer_skip_1_upsampled, layer3_enc_out)
    
    # Upsample again to get the last layer
    dnn_last_layer = tf.layers.conv2d_transpose(layer_skip_2, num_classes, 4,
                                                strides=(2,2),
                                                padding='same',
                                                kernel_initializer=kernel_Init,
                                                kernel_regularizer=kernel_Regu)
    
    return dnn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_operation = optimiser.minimize(cross_entropy_loss) 
    
    return logits, cross_entropy_loss, train_operation
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    print("Training ... ")
    print("=============")
    print()
    
    for i in range(epochs):
        print("EPOCH {} ...")
        print()
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                    input_image: image, correct_label: label, keep_prob: 0.5})
            print("Loss :   {:.3f}".format(loss))
            print()
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        EPOCHS = 50
        BATCH_SIZE = 5
        
        # Create Placeholder Variables
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes],
                                       name='correct_label')
        learning_rate = tf.constant(0.0001)
        
        # Get VGG layers with its weights out
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        # Create the new connections using FCN
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        #Get logits, cross entroy and train operation
        logits, cross_entropy_loss, train_operation = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_operation,
                 cross_entropy_loss, input_image, correct_label, keep_prob,
                 learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
