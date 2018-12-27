cd utils/detection
cp *.py *.pkl models/research

cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python extract_detection_feature.py
