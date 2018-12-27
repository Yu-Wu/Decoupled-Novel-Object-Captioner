# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-08-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
from datetime import datetime

from feature_extractor.feature_extractor import FeatureExtractor
import feature_extractor.utils as utils

import six.moves.urllib as urllib
import tarfile, os
import multiprocessing as mp


Downloads = {"vgg_16": "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"}



def feature_extraction_queue(feature_extractor, image_path, layer_names,
                             batch_size, num_classes, num_images=200000):
    '''
    Given a directory containing images, this function extracts features
    for all images. The layers to extract features from are specified
    as a list of strings. First, we seek for all images in the directory,
    sort the list and feed them to the filename queue. Then, batches are
    processed and features are stored in a large object `features`.

    :param feature_extractor: object, TF feature extractor
    :param image_path: str, path to directory containing images
    :param layer_names: list of str, list of layer names
    :param batch_size: int, batch size
    :param num_classes: int, number of classes for ImageNet (1000 or 1001)
    :param num_images: int, number of images to process (default=100000)
    :return:
    '''

    # Add a list of images to process, note that the list is ordered.
    image_files = utils.find_files(image_path, ("jpg", "png"))
    num_images = min(len(image_files), num_images)
    image_files = image_files[0:num_images]

    num_examples = len(image_files)
    num_batches = int(np.ceil(num_examples/batch_size))

    # Fill-up last batch so it is full (otherwise queue hangs)
    utils.fill_last_batch(image_files, batch_size)

    print("#"*80)
    print("Batch Size: {}".format(batch_size))
    print("Number of Examples: {}".format(num_examples))
    print("Number of Batches: {}".format(num_batches))

    # Add all the images to the filename queue
    feature_extractor.enqueue_image_files(image_files)

    # Initialize containers for storing processed filenames and features
    feature_dataset = {'filenames': []}
    for i, layer_name in enumerate(layer_names):
        layer_shape = feature_extractor.layer_size(layer_name)
        layer_shape[0] = len(image_files)  # replace ? by number of examples
        feature_dataset[layer_name] = np.zeros(layer_shape, np.float32)
        print("Extracting features for layer '{}' with shape {}".format(layer_name, layer_shape))

    print("#"*80)

    # Perform feed-forward through the batches
    for batch_index in range(num_batches):

        t1 = time.time()

        # Feed-forward one batch through the network
        outputs = feature_extractor.feed_forward_batch(layer_names)

        for layer_name in layer_names:
            start = batch_index*batch_size
            end   = start+batch_size
            feature_dataset[layer_name][start:end] = outputs[layer_name]

        # Save the filenames of the images in the batch
        feature_dataset['filenames'].extend(outputs['filenames'])

        t2 = time.time()
        examples_in_queue = outputs['examples_in_queue']
        examples_per_second = batch_size/float(t2-t1)

        if batch_index % 10 == 0:
            print("[{}] Batch {:04d}/{:04d}, Batch Size = {}, Examples in Queue = {}, Examples/Sec = {:.2f}, Feature Dim {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), batch_index+1,
                num_batches, batch_size, examples_in_queue, examples_per_second, layer_shape[-1]
            ))

        save_queue.put([outputs['filenames'], outputs[layer_name]])

    # If the number of pre-processing threads >1 then the output order is
    # non-deterministic. Therefore, we order the outputs again by filenames so
    # the images and corresponding features are sorted in alphabetical order.
    if feature_extractor.num_preproc_threads > 1:
        utils.sort_feature_dataset(feature_dataset)

    # We cut-off the last part of the final batch since this was filled-up
    feature_dataset['filenames'] = feature_dataset['filenames'][0:num_examples]
    for layer_name in layer_names:
        feature_dataset[layer_name] = feature_dataset[layer_name][0:num_examples]

    return feature_dataset




def save_results_thread(save_dir, save_queue):
    while True:
        try:
            image_paths, feats = save_queue.get(True, 60)
        except Exception as e:
            print("Quit save_results_thread.")
            return

        for image_path, feat in zip(image_paths, feats):
            image_path = str(image_path, encoding = "utf-8")
            name = image_path[image_path.rfind('/')+1:]
            save_feat_path = os.path.join(save_dir, name)
            np.savez(save_feat_path, cnn_feat=feat[0][0])



################################################################################
################################################################################
################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TensorFlow feature extraction")
    parser.add_argument("--network", dest="network_name", type=str, required=True, help="model name, e.g. 'resnet_v2_101'")
    parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", type=str, default="./cnn_feature_extraction/checkpoints", help="path to pre-trained checkpoint file")
    parser.add_argument("--image_path", dest="image_path", type=str, required=True, help="path to directory containing images")
    parser.add_argument("--save_dir", dest="save_dir", type=str, required=True, help="path to save features")
    parser.add_argument("--layer_names", dest="layer_names", type=str, required=True, help="layer names separated by commas")
    parser.add_argument("--preproc_func", dest="preproc_func", type=str, default=None, help="force the image preprocessing function (None)")
    parser.add_argument("--preproc_threads", dest="num_preproc_threads", type=int, default=2, help="number of preprocessing threads (2)")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="batch size (32)")
    parser.add_argument("--num_classes", dest="num_classes", type=int, default=1000, help="number of classes (1001)")
    args = parser.parse_args()

    network_name = args.network_name
    ckpt_path = os.path.join(args.checkpoint_dir, network_name + ".ckpt")
    print("Checkpoint path:", ckpt_path)

    # if checkpoint not found, download the pretrained model from tensorflow official repo.
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found!\nBegin to download from", Downloads[network_name])
        opener = urllib.request.URLopener()
        opener.retrieve(Downloads[network_name], network_name)
        tar_file = tarfile.open(network_name)
        tar_file.extractall(path=args.checkpoint_dir)
        os.remove(network_name)
    
    if not os.path.exists(args.save_dir):
        print("Creating Save Dir ", args.save_dir)
        os.makedirs(args.save_dir)
    else:
        print("Save Dir '{}' Exists".format(args.save_dir))


    # resnet_v2_101/logits,resnet_v2_101/pool4 => to list of layer names
    layer_names = args.layer_names.split(",")

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(
        network_name=network_name,
        checkpoint_path=ckpt_path,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        preproc_func_name=args.preproc_func,
        preproc_threads=args.num_preproc_threads
    )


    save_queue = mp.Queue()
    p = mp.Process(target=save_results_thread, args=(args.save_dir, save_queue)); p.start()


    # Print the network summary, use these layer names for feature extraction
    #feature_extractor.print_network_summary()

    # Feature extraction example using a filename queue to feed images
    image_paths = [os.path.join(args.image_path, "train2014"), os.path.join(args.image_path, "val2014")]
    feature_dataset = feature_extraction_queue(
        feature_extractor, image_paths, layer_names,
        args.batch_size, args.num_classes)

    # Write features to disk as HDF5 file
    #utils.write_hdf5(args.out_file, layer_names, feature_dataset)
    #print("Successfully written features to: {}".format(args.out_file))

    # Close the threads and close session.
    feature_extractor.close()
    p.join()
    print("Finished.")
