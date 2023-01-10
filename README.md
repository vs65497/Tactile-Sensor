# Tactile-Sensor
Manipulation is an important aspect of modern robotics. This field focuses on grasping, picking, and placing objects. Grasping a sharp, slippery, soft, or strangely shaped object can prove difficult. While this provides little challenge to humans, many robots struggle without a sense of touch. As a result researchers have developed tactile sensors. In effect, their use allows for more effective picking and placing configurations -- especially when used in combination with vision.

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/intro.png" height="400"><br />_Image_: https://spectrum.ieee.org/why-tactile-intelligence-is-the-future-of-robotic-grasping 

# Experimental Setup and Purpose
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/experimental_setup.png" width=600><br />_TPU finger (left). PLA case (right) allows for repeatable deformations at the same location and prevents warping of the finger._

**Purpose**: To make a vision-based tactile sensor which converts images of detected deformations to data with location and depth. Vision-based means applying computer vision with a camera (iPhone 14) aimed at the inside surface of a TPU dome. This experiment takes inspiration from talks given by Katherine Kuchenbecker and Oliver Brock.

# Experimental Results
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/results_top.png" width=600><br />
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/results_bottom.png" width=600>

The top photo is taken from test footage similar to [this video](https://youtube.com/shorts/2dx2I3SYDVk). It contains a deformation caused by pressing a pen into the side of the TPU finger. Below (_left_), the pipeline has identified the deformation as a red dot on this unrolled version of the cone. The depth of the deformation is estimated to be 3.36 mm. The data discovered from object detection and depth estimation (_right_).

# Procedure
With an understanding of the pipeline's input and output, let’s discuss the process by which this is done. Below is an infographic showing the full procedure of training and executing the pipeline.

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/Tactile Sensor Pipeline.png"><br />
_Pipeline for developing the tactile sensor. Includes 3 phases: YOLOv5, DNN, and Live Data -> Visualization._

## Phase 1: YOLOv5
**_Objective_**: Fine-Tune YOLOv5 Model to obtain bounding box data on deformations.<br />
**_Notebook_**: https://github.com/zanzivyr/Tactile-Sensor/blob/main/YOLOv5_Deformations_Training.ipynb <br />
**_Test Footage_**: https://youtube.com/shorts/2dx2I3SYDVk

1. **Conduct Experiment** - A video is recorded with several deformations at different angles and depths being made. Care is taken not to warp the finger or move the camera.
2. **Video Data** - This video is not a live stream, it is saved.
3. **Select Photos from Video Data** - Several still frames are taken from the video showcasing a variety of different deformations. All data can be found on Roboflow - https://app.roboflow.com/tactile-sensor/deformations/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true 
4. **Annotate Bounding Boxes** - Using https://www.makesense.ai/ we annotate bounding boxes for training. The photos and labels are then uploaded to Roboflow allowing users to make different splits between training, test, and validation data sets. Afterward other preprocessing and augmentation steps are available for normalization and  synthetic data. Without additional video we can generate 4x more data for training.

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/processing_top.png" width=600><br />
_Roboflow Preprocessing and Augmentation steps (left). Spread of bounding boxes over normalized data (right). Notice the gap at the bottom - the iPhone light created a bright highlight exactly here making detections nearly impossible._
<br /><br />

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/processing_bottom.png" width=600><br />
_Augmented data set (left). Detected deformation (right)._

5. **Fine-Tune YOLOv5** - Using a small amount of data and transfer learning, we fine-tune YOLOv5 to detect deformations. Accurate detections were obtained with 120 epochs of 16 batches.

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/processing2.png" width=600><br />
_Roboflow was able to successfully detect deformations based on my training data. However, Roboflow’s trained model is not used in this experiment._

## Phase 2: Deep Neural Network (DNN)
**_Objective_**: Train a DNN to predict deformation depth.<br />
**_Notebook_**: https://github.com/zanzivyr/Tactile-Sensor/blob/main/Tactile_Sensor_CNN.ipynb 

1. **Object Detection** - Using the previously trained YOLOv5 model.
2. **Deformations** - A tensor of detected deformations is created.
3. **Annotate Depths** - Given these deformations, we manually annotate depths and add the data to the tensor.
4. **Reshape Tensor** - Adds more fields to the tensor to give the next neural network more features to train on.
5. **Normalize Data** - This gives the neural network a battery ability to compare each data point.
6. **Train Deep Neural Network (DNN)** - Using a DNN with a single fully connected layer, with 100 neurons, we pass a tensor with 9 features in and expect to receive one output - _depth_. Hyperparameters and parameters are not tuned for this experiment.

<img src="https://qph.cf2.quoracdn.net/main-qimg-6f8d8e883d420ae86036f0e2a00f4161-lq" width=400><br />
_Image_: https://www.quora.com/Why-dont-we-initialize-the-weights-of-a-neural-network-to-zero<br />
_Deep Neural Network with 1 hidden layer. Our DNN has 9 features in the input and 100 neurons in the hidden layer._

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/tensor.png" width=500><br />
_A tensor with 9 features. The 10th column, depth, is excluded for training._

## Phase 3: Live Data, Visualize
**_Objective_**: Convert live data into a 2D visualization with usable data.<br />
**_Notebook_**: https://github.com/zanzivyr/Tactile-Sensor/blob/main/Tactile_Sensor_Visualization.ipynb 

1. **Object Detection** - Same as Phase 2.
2. **Partial Deformation Data** - This gives the 9 feature tensor without depth.
3. **Reshape Tensor** - Same as Phase 2.
4. **Normalize Data** - Same as Phase 2.
5. **Depth Inference** - We feed the normalized deformation data into the DNN to obtain a depth inference.
6. **Full Deformation Data** - This gives a 10 feature tensor including depth.
7. **De-normalize Data** - We convert the tensor back to its real values given normalization information saved during training of the DNN.
8. **Image Coordinates to Physical Coordinates** - The image from the camera is a 2D image which includes warping from properties of the lens and optics. In this experiment we do not reverse these optics, but rather provide an approximation based on relative size of the small diameter of the top of the cone.
9. **Unroll Truncated Cone** - The 3D truncated cone is unrolled into a 2D form for visualization.
10. Place Deformation on 2D Plane.

# Conclusion
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/initial_concept.png" width=600><br />
_Initial sketch of how data for training (left). Katherine Kuchenbecker’s sensor (right). The majority of this experiment is aimed at reverse engineering this one slide._

This experiment was a success in converting image data into tensor data. Ideally, given this 2D visualization data, it is now possible to do discover other dynamics such as gripping forces, orientation, and slipping by tracking position and depth over time. However, this experiment is limited in that it only processes still images at the moment. Some work would need to be done to allow saved video and then streamed video. The visualization format is also less than desired. Initially visualization was intended to show 2D heatmaps and a 3D Digital Twin.

Vision detection can be improved by using dots on the inside of the TPU finger, green and red lights emitting from opposite sides of the sensor, and changing the shape of the sensor to something more conducive to gripping.

## References

Those two conference talks and their related papers
_MIT Robotics - Katherine Kuchenbecker - Tactile Sensing for Robots with Haptic Intelligence_
- https://youtu.be/0Vg0jkzVaFw?t=1283
- https://arxiv.org/abs/1511.06065 
_MIT Robotics - Oliver Brock - Why I Believe That AI-Robotics is Stuck_
- https://youtu.be/tr6aatJL84A?t=1440 

_FastAI fastbook and any related papers_
- https://github.com/fastai/fastbook
- https://docs.fast.ai/tutorial.tabular.html 
- https://arxiv.org/pdf/1604.06737.pdf
- https://arxiv.org/pdf/1606.07792.pdf 

_YOLO description, tutorial and the original paper_
- https://medium.com/analytics-vidhya/yolo-explained-5b6f4564f31
- https://www.v7labs.com/blog/yolo-object-detection
- https://towardsdatascience.com/yolo-v5-object-detection-tutorial-2e607b9013ef 
- https://arxiv.org/pdf/2209.02976.pdf 

_Why Tactile Intelligence Is the Future of Robotic Grasping_
- https://spectrum.ieee.org/why-tactile-intelligence-is-the-future-of-robotic-grasping 

_Why don't we initialize the weights of a neural network to zero?_
- https://www.quora.com/Why-dont-we-initialize-the-weights-of-a-neural-network-to-zero
