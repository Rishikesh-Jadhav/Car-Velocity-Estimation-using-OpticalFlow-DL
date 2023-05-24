# RAFT
The package was made for ENPM 673 final project by group 13. The package is based on https://github.com/princeton-vl/RAFT. 
# dependencies
test2.yml file of the conda environment is provided.  
The code has been tested with 
# Yolo Weights
The weights for yolo need to be downloaded and need to be placed in the root folder.
## Demos
For running the demo using our trained dataset. 
-python demo.py --model=models/enpm673_raft-kitti.pth --path=frames
For running with videos.
-python inference.py --model=models/enpm673_raft-kitti.pth
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/enpm673_raft-kitti.pth --dataset=kitti --mixed_precision
```
## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```
If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.

Other files and folder:
- weights and cfg files for yolo were obtained from https://github.com/pjreddie/darknet 
- Yolo weights can be downloaded from https://drive.google.com/drive/folders/1h9PqeZ3l5RUURJIxeNMXGt-Ilil3ngrO?usp=sharing, I got this from here https://pjreddie.com/media/files/yolov3.weights
- 1684366668.0944755.mp4 video for testing
- stream.py- for streaming the video from the pi cam from https://singleboardblog.com/real-time-video-streaming-with-raspberry-pi/ 

Note: 
- To visualize the motion vectors of the car uncomment line 51-56 in demo.py and 62-68 for inference.py
- To visualize the bounding rectangle of the car uncomment line 133-137 in demo.py and 132-135 for inference.py
- for live video streaming hardware is needed but we used line 210 of inference.py to do so. The IP address will change. 
* We downloaded images online and annotated them for yolo training and testing but we could not get right output so we switched to 
