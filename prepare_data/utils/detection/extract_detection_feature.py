import sys
import os
import numpy as np
import six.moves.urllib as urllib
import tarfile

from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2

from google.protobuf import text_format


import tensorflow as tf
import h5py
from PIL import Image
import re, tqdm, random
import pickle as pkl
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


import multiprocessing as mp
import re, tqdm, time


# Note the current path is $PWD/prepare_data/detection/models/research
IMAGE_DIR = '../../../mscoco/'
SAVE_DIR = os.path.join(IMAGE_DIR, "extracted_object_memory_17_11_08")

MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'


PATH_TO_CKPT = 'object_detection/' + MODEL_NAME + "/"
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
visual = False


MAX_CANDIDATES = 10


if not os.path.exists(PATH_TO_CKPT):
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    print("Do not find the pre-trained model in " + PATH_TO_CKPT)
    print("Begin to download pre-trained models from " + DOWNLOAD_BASE + MODEL_FILE)
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

    tar_file = tarfile.open(MODEL_FILE)
    tar_file.extractall(path='./object_detection/')


def read_image(image_path):  
    image = Image.open(image_path)
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    new_image = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)
    return new_image




def read_image_np_thread(image_path_queue, img_np_queue):  
    while not image_path_queue.empty():
        try:
            video_imgs = []
            image_path = image_path_queue.get(True, 10)

            image = Image.open(image_path)
            image = image.convert('RGB')
            (im_width, im_height) = image.size
            new_image = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)
            img_np_queue.put([image_path, new_image])
        except Exception as e:
            print("quit read_image_np_thread. ", e)
            return 

def save_result(image_path, det_classes, det_feature):
    name = image_path[image_path.rfind('/')+1:]
    save_feat_path = os.path.join(SAVE_DIR, name)

    np.savez(save_feat_path, det_feature=det_feature, det_classes=det_classes)



def main():
    image_path_queue = mp.Queue()
    img_np_queue = mp.Queue()

    # collect total images for detection
    total_images = 0
    for dir_name in ['train2014', 'val2014']:
        dir_path = os.path.join(IMAGE_DIR, dir_name)
        for image_name in sorted(os.listdir(dir_path)):
            image_path = os.path.join(dir_path, image_name)
            image_path_queue.put(image_path)
            total_images += 1
    print("Need to Detect:", total_images, "Images.")


    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        print("Found existing dir in", SAVE_DIR)
        #print("Quit")
        #return

    p1 = mp.Process(target=read_image_np_thread, args=(image_path_queue, img_np_queue));p1.start()


    # get graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      coco_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT+"/frozen_inference_graph.pb", 'rb') as fid:
        serialized_graph = fid.read()
        coco_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(coco_graph_def, name='')


    # get feature and predict
    name = "FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/add"
    extract_feature_operation = detection_graph.get_operation_by_name(name)
    name = "SecondStageBoxPredictor/ClassPredictor/BiasAdd"
    class_predict_logits_op = detection_graph.get_operation_by_name(name)


    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # def box feature operation
        box_ind = tf.constant(0, shape=[MAX_CANDIDATES])
        box_feature_op = tf.image.crop_and_resize(extract_feature_operation.outputs[0], 
                                               detection_boxes[0][0:MAX_CANDIDATES], 
                                               box_ind,
                                               crop_size = [17, 17])

        box_feature_op = tf.reduce_mean(box_feature_op, axis=[1,2])


        with tqdm.tqdm(total=total_images) as pbar:
            
            begin = time.time()
            for idx in range(total_images):
                image_path, image_np_expanded = img_np_queue.get(True, 60)

                (det_classes, det_feature) = sess.run(
                  [detection_classes, box_feature_op],
                  feed_dict={image_tensor: image_np_expanded})

                det_classes = det_classes[0][:MAX_CANDIDATES]

                save_result(image_path, det_classes, det_feature)

                if idx % 100 == 0:
                    tqdm.tqdm.write("Extracted {} images".format(idx))
                pbar.update(1)



if __name__ == "__main__":
	main()    


