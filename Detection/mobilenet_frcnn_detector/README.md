# MobileNet+FRCNN for traffic sign detection
Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.

Mobilenet based traffic sign detctor

NOTE:
- We trained and evalute our detector on GTSDB dataset.
- We evaluated the trained mobilenet detector directly on Tsingshua-Tencent 100K (no model refinement )


USAGE:
- Tensorflow backend is highly recommended.
- `train_frcnn.py` is not released yet, we only release `train_frcnn.py` for now
- Original training data format
   
   line containing:

    `filepath,x1,y1,x2,y2,class_name`

    For example:

    /data/imgs/img_001.jpg,837,346,981,456,cow
    
    /data/imgs/img_002.jpg,215,312,279,391,cat

    The classes will be inferred from the file. To use the simple parser instead of the default pascal voc style parser,
    use the command line option `-o simple`. For example `python train_frcnn.py -o simple -p my_data.txt`.

- Running `train_frcnn.py` will write weights to disk to an hdf5 file, as well as all the setting of the training run to a `pickle` file. These
settings can then be loaded by `test_frcnn.py` for any testing.

- test_frcnn.py can be used to perform inference, given pretrained weights and a config file. Specify a path to the folder containing
images (run under the mobilenet_frcnn_detecotor Directory):
    
    `python test_frcnn.py -p /path/to/test_data/ -n 512`
- Data augmentation can be applied by specifying `--hf` for horizontal flips, `--vf` for vertical flips and `--rot` for 90 degree rotations

- train command：  

  `python train_frcnn.py -o simple -p data/my_data.txt -n 300 --input_weight_path weight/mobilenet_1_0_224_tf_no_top.h5 --hf --network mobilenet --num_epochs 200 --output_weight_path mobilenet.hdf5
`


- mAP: measure map

`python measure_map.py -p data/my_data.txt -o simple -n 64`


Acknowledgments:
Code borrows heavily from [keras-frcnn] (https://github.com/yhenon/keras-frcnn)
