# YOLOv3_DeepSORT

An implementation of real-time object detection and tracking using [YOLOv3](https://arxiv.org/abs/1804.02767) 
and [Deep SORT](https://arxiv.org/abs/1703.07402). The script can work either with the web camera or with a video file.

**Use [this link](https://pjreddie.com/media/files/yolov3.weights)
to download *yolov3.weights*, and place the file in the "yolov3" folder.**

If the [requirements for TensorFlow to run on GPU](https://www.tensorflow.org/install/gpu) are met, the Deep SORT 
portion of the script will use the GPU for processing. The YOLOv3 with OpenCV implementation will run on the CPU, but 
[this guide](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) 
can be followed to enable OpenCV's "dnn" module to use an NVIDIA GPU instead, resulting in a significant speed improvement.

---
This project adds Deep SORT tracking to [this previous YOLOv3 with OpenCV implementation.](https://github.com/BorislavY/YOLOv3_OpenCV)
Detailed comments were added to show understanding of the code.

The files found in the "deep_sort" folder are taken from the [original Deep SORT implementation](https://github.com/nwojke/deep_sort) 
made by the authors of the paper. Some of the files have been altered to add class name associations for the tracks.
Class names smoothing has been implemented to avoid very frequent changes of class names. 
The effectiveness of this can be seen in the example videos below. 


### Video processed with no class names smoothing:
![GIF missing :(](results1.gif)

### Video processed with class names smoothing of 60 frames:
![GIF missing :(](results2.gif)