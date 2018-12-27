## Preparing Data 

Please follow the four scripts to download and preprocess the MSCOCO data.

### Download MSCOCO Image Data
```shell
sh step1_download_coco.sh
```
The script will download the image data from the MSCOCO official site and extract it to `mscoco`.

### Generating Detection Results
```shell
sh step2_detection.sh
```
We use the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to prepocess each image and save the detection results to disk. 
Note that it may takes about 12 hours to finish the preprocessing with a Nvidia V100 GPU.


### Extract Image Feature
```shell
sh step3_image_feature_extraction.sh
```
It may takes 20 minutes to finish the feature extraction process.

### Generate the NOC COCO Dataset
```shell
sh step4_transfer_coco_to_noc.sh
```
We transfer the original MSCOCO dataset to fit the novel object captioning setting.

All the preprocessed results can be found in `mscoco`. 
