# Tactile-Sensor
Manipulation is an important aspect of modern robotics. This field focuses on grasping, picking, and placing objects. Grasping an object can be difficult because of its material properties – it could be sharp, slippery, soft, or strangely shaped. For humans, this really provides little challenge because of our skin, however, most current robots have no way of sensing these properties aside from vision. Because of this, researchers have been investigating methods for grippers to detect surface features of the target object. Knowing the shape of an object allows for choosing better picking configurations and more effective placing, especially when used in combination with vision. A tactile sensor helps to fulfill this need.

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/intro.png" height="400">
https://spectrum.ieee.org/why-tactile-intelligence-is-the-future-of-robotic-grasping 

# Experimental Setup
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/experimental_setup.png" width=600><br />_TPU finger (left). PLA case (right) allows for repeatable deformations at the same location and prevents warping of the finger._

# Results
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/results_top.png" width=600>
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/results_bottom.png" width=600>

Let’s have a look at the results first then explain how we got here. At the top is a photo taken from test footage similar to this: https://youtube.com/shorts/2dx2I3SYDVk. It contains a deformation caused by pressing a pen into the side of the TPU finger. Below (_left_), the pipeline has identified the deformation and is showing it as a red dot on this unrolled version of the cone. The depth of the deformation is estimated to be 3.36 mm. Next to it (_right_), is the data discovered from object detection and depth estimation.

# Procedure
Now that we’ve seen the beginning and end of the pipeline, let’s discuss the details of how this is done. Below is an infographic showing the full procedure of training and using the pipeline.

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/Tactile Sensor Pipeline.png">

## Phase 1: YOLOv5
**_Objective_**: Fine-Tune YOLOv5 Model to obtain bounding box data on deformations.
**_Notebook_**: https://github.com/zanzivyr/Tactile-Sensor/blob/main/YOLOv5_Deformations_Training.ipynb 
**_Test Footage_**: https://youtube.com/shorts/2dx2I3SYDVk

1. **Conduct Experiment** - A video is recorded with several deformations at different angles and depths being made. Care is taken not to warp the finger or move the camera.
2. **Video Data** - This video is not a live stream, it is saved.
3. **Select Photos from Video Data** - Several still frames are taken from the video showcasing a variety of different deformations. All data can be found on Roboflow - https://app.roboflow.com/tactile-sensor/deformations/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true 
4. **Annotate Bounding Boxes** - Using the website https://www.makesense.ai/ we annotate bounding boxes for training. These photos and labels are then uploaded to Roboflow which then allows users to make different splits between training, test, and validation data sets. Users are also able to add other preprocessing and augmentation steps which are important for normalization and creating synthetic data. With this we are able to 4x the sample data and make training more effective.

<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/processing_top.png" width=600>
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/processing_bottom.png" width=600><br />_Roboflow Preprocessing and Augmentation steps (top left). Spread of bounding boxes over normalized data (top right). Notice that the data had a gap at the bottom - the light from the iPhone created a bright spot exactly in that gap making detections there nearly impossible. Augmented data set (bottom left). Detected deformation (bottom right)._

5. **Fine-Tune YOLOv5** - Using a small amount of data and transfer learning, we fine-tune YOLOv5 to detect deformations. I was able to get accurate detection with a large amount of epochs (120) and a few batches (16).
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/processing2.png" width=600><br />_Roboflow was able to successfully detect deformations based on my training data. However, I did not use Roboflow’s model in this experiment._

## Phase 2: Deep Neural Network (DNN)
**_Objective_**: Train a DNN to predict deformation depth.
**_Notebook_**: https://github.com/zanzivyr/Tactile-Sensor/blob/main/Tactile_Sensor_CNN.ipynb 

1. **Object Detection** - Using the previously trained YOLOv5 model.
2. **Deformations** - A tensor of detected deformations is created.
3. **Annotate Depths** - Given these deformations, we manually annotate depths and add the data to the tensor.
4. **Reshape Tensor** - Add more fields to the tensor to give the next neural network more features to train on.
5. **Normalize Data** - This gives the neural network a battery ability to compare each data point.
6. **Train Deep Neural Network (DNN)** - Using a DNN with a single fully connected layer, with 100 neurons, we pass a tensor with 9 features in and expect to receive one output - _depth_. I did not tune hyperparameters for this experiment. And I did not experiment with parameters due to time constraints.
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/tensor.png" width=700>
A tensor with 9 features. The 10th column, depth, is excluded for training.

## Phase 3: Live Data, Visualize
**_Objective_**: Convert live data into a 2D visualization with usable data.<br />
**_Notebook_**: https://github.com/zanzivyr/Tactile-Sensor/blob/main/Tactile_Sensor_Visualization.ipynb 

1. **Object Detection** - Same as Phase 2
2. **Partial Deformation Data** - This gives the 9 feature tensor without depth.
3. **Reshape Tensor** - Same as Phase 2
4. **Normalize Data** - Same as Phase 2
5. **Depth Inference** - We feed the normalized deformation data into the DNN to obtain a depth inference.
6. **Full Deformation Data** - This gives a 10 feature tensor including depth.
7. **De-normalize Data** - We convert the tensor back to its real values given normalization information saved during training of the DNN.
8. **Image Coordinates to Physical Coordinates** - The image from the camera is a 2D image which includes warping from properties of the lens and optics. In this experiment we do not reverse these optics, but rather provide an approximation based on relative size of the small diameter of the top of the cone.
9. **Unroll Truncated Cone** - The 3D truncated cone is unrolled into a 2D form for visualization.
10. Place Deformation on 2D Plane.

# Initial Concept
<img src="https://github.com/zanzivyr/Tactile-Sensor/blob/main/presentation/initial_concept.png" width=600><br />_Initial sketch of how data for training (left). Katherine Kuchenbecker’s sensor (right). The majority of this experiment is aimed at reverse engineering this one slide._

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

Why Tactile Intelligence Is the Future of Robotic Grasping
- https://spectrum.ieee.org/why-tactile-intelligence-is-the-future-of-robotic-grasping 
