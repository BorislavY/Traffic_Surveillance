# Traffic Surveillance

An implementation of real-time traffic surveillance through object detection and tracking using [YOLOv3](https://arxiv.org/abs/1804.02767) 
and [Deep SORT](https://arxiv.org/abs/1703.07402). The script can work either with a camera feed or with a video file.
Detailed comments are added to show a good understanding of the code.

**Use [this link](https://pjreddie.com/media/files/yolov3.weights)
to download *yolov3.weights*, and place the file in the "yolov3" folder.**

If the [requirements for TensorFlow to run on GPU](https://www.tensorflow.org/install/gpu) are met, the Deep SORT 
portion of the script will use the GPU for processing. The YOLOv3 with OpenCV implementation will run on the CPU, but 
[this guide](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) 
can be followed to enable OpenCV's "dnn" module to use an NVIDIA GPU instead, resulting in a significant speed improvement.

The files found in the "deep_sort" folder are taken from the [original Deep SORT implementation](https://github.com/nwojke/deep_sort) 
made by the authors of the paper. Some of the files have been altered to add necessary functionality for this task.

---

The script is currently configured to work on a video of an intersection, logging into a CSV file each vehicle's Id, 
type, name of the road that the vehicle approached the intersection from, and the name of the road that the
vehicle exited the intersection from. Pictures of each detected vehicle are also stored in the "counted_vehicles"
folder in the format: type_Id.png

The detailed comments allow for the script to be easily reconfigured for other traffic surveillance tasks, 
such as counting the number of vehicles that pass in each direction of a highway. It can even be configured
for surveillance of other objects, such as pedestrians.

---

Since the original pre-trained YOLOv3 and DeepSORT models are used, the object detection and tracking are not of the
highest accuracy, but this can be improved by training the models for the particular task.

Another main issue which is yet to be resolved is that occlusions can cause the algorithm to lose track of an object 
for a brief period, and if this happens while the object is passing over a line segment of interest used for 
counting/logging, the object won't be counted/logged.

## Example Output

__The video being processed:__

![GIF missing :(](results.gif)

__Resulting CSV file:__

![Image missing :(](https://i.imgur.com/zqxE7E1.png)

__Captured images:__

![Image missing :(](https://i.imgur.com/RrLutdC.png)

__Annotations of the same video made by a real person (CSV only):__

![Image missing :(](https://i.imgur.com/8mQsptr.png)

All of the entries made by the algorithm are correct, but a few vehicles weren't logged due to the algorithm losing
track of them while they were passing over a line of interest representing one of the roads. The problem can be solved by improving the 
tracking and detection algorithms.