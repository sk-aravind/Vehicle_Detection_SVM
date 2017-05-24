[//]: # (Image References)
[image1]: ./readme_imgs/cars.png
[image2]: ./readme_imgs/notcars.png
[image3]: ./readme_imgs/hog.png
[image4]: ./readme_imgs/pipeline.png
[image6]: ./readme_imgs/detection.png
[image7]: ./readme_imgs/final_box.png
[image8]: ./readme_imgs/colorspace.png
[image7]: ./readme_imgs/slidingwindow.png
[video1]: ./readme_imgs.mp4

# __Vehicle Detection Using Linear SVMs__


## Code Structure
All the code for this implementation is contained in the python jupyter notebook and the code cells are labelled as follows:

1. Imports
2. Load Test & Dataset Images
  1. Dataset Visualization
3. Feature Extraction Pipeline
  1. HOG Visualized
  2. Exploring Colorspaces
  3. Extract Features
4. Prepare Data
5. Training SVC Classifier
  1. Save Data to Pickle
6. Camera Calibration
7. Subsampling HOG Detector
  1. Visualize Sliding Windows
8. Heatmap
9. Debugging View
10. Temporal Heatmap
11. Overall Pipeline
12. Process Video





# __Training Pipeline__

## 1) Dataset Summary

The data set comprises of 2 categories of 64 x 64 RGB images. The categories being vehicles and non-vehicles. Each category has approximately 8500 images each.

#### __Vehicle Data__
![alt text][image1]

#### __Non-vehicle Data__
![alt text][image2]

## 2) Extracting Features from Images
### 2.1) Histogram of Oriented Gradients (HOG)

The code for HOG Feature Extraction can be found in the cell labeled  __3) Feature Extraction Pipeline__

###### __AIM__
To extract edge features of the car using histograms of oriented gradients. The features should be able to generalize well over a variety of colors and varying perspectives.

###### __APPROACH__

To attain the HOG-representation from a RGB image these steps were implemented.

* Convert RGB to YCrCb colorspace
* Use the function 'skimage.hog()' to convert all color 3 channels into the 3 respective hog images


Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

###### __PARAMETER TUNING__
After exploring different color spaces and using different number of channels and with `skimage.hog()`. I choose to use all the YCrCb color channels as it resulted in the highest accuracy in the classifier.

Besides tuning the parameters to only optimize the accuracy, we also have to consider the computational complexity of the resulting parameters. As there is a point where a marginal increases in accuracy results in more costly computations. For this reason I choose a grid size of 8x8 ( 8 pixels per cell) and 2x2 cells per block for a 64x64 image. As reducing the parameters further results in a loss of information and accuracy.




### 2.2) Spatial Features

The code for Spatial Feature Extraction can be found in the cell labeled  __3) Feature Extraction Pipeline__

###### __AIM__
To extract raw features of the car based on pixel values. To capture the color,spatial signature of a car.

###### __APPROACH__
* Resize image to smaller size
* Unravel image into feature vector

###### __PARAMETER TUNING__
In order to keep the number of features low and maintain the generalization of this features we resize the images to 32x32 and unravel the image in a feature vector. Spatial binning alone does not result in a robust classifier, however when used along with HOG features and color histograms it improved the accuracy considerably.

### 2.3) Color Features

The code for Color Feature Extraction can be found in the cell labeled  __3) Feature Extraction Pipeline__


![alt text][image8]

###### __AIM__
To extract color features of the car based on pixel values. To capture the color signature of a car.

###### __APPROACH__
* Split into 3 separate color channels
* Allocate color distribution into bins
* Concatenate all 3 into a feature vector


## 3) Training Classifier

The code for training the classifier can be found in the cell labeled  __5) Training SVC Classifier__

The HOG, color histogram and spatial features were combined into a single 1-D feature vector. Features were normalized using `sklearn.preprocessing.StandardScaler()`. 80% of the data was split and used to train the SVM and 20% was allocated to the test set.

The parameters used to train the SVM :
* colorspace = YCrCb
* orientations = 9
* pixels per cell = (8,8)
* cells per block = (2,2)
* color histogram bins = 32 per color channel
* spatial bins = (32,32)

With these parameters the classifier averaged around 98-99% accuracy on the test set. Which was a satisfactory balance of accuracy and performance. As adding more features to the features vector only resulted in marginal accuracy improvements.

## 4) Sliding Window Search

The code for the sliding window search can be found in the cell labeled  __7) Subsampling HOG Detector__

Utilizing a sliding window approach we can search for vehicles in regions of interest. In this case the region of interest is the road, the bottom half of the image. Since vehicles that are farther away appear smaller in the image than vehicles that are near the camera to will search at different scales for different portions of the road. Due to the perspective of the camera these portions will overlap each other.

These specific set of scales and areas were chosen for each part of the road, in order to frame the car in similar proportions to the car image examples in the training data.

![alt text][image7]

So we run the classifier on each of the sliding windows and we will obtain multiple positive detections as seen in this image. We can greatly reduce computational cost by negating certain search areas such as the non-drivable areas to the left of the car.


---

## Video Implementation

#### Link to Final Result
ADD GIF
Here's a [link to my video result](./project_video.mp4)


### Temporal Heatmap

I recorded the positions of positive detections in each frame of the video and added each detection to a list. Each frame produces one such list, in order to implement a temporal heatmap, I stored n number of such list and created a heatmap from these combined list. I then thresholded that map to identify vehicle positions. By using the  `scipy.ndimage.measurements.label()` I was able identify individual entities in the heatmap.  I then assumed each entity corresponded to a vehicle.  I constructed bounding boxes to cover the area of each detected entity.  

This step helped to remove false positives that would be mis-classified in a few consecutive frames and also helps to smooth the tracking of the bounding box.

Below is the visualization of the detection and smoothing pipeline, the result of thresholded temporal heatmap is passed to `scipy.ndimage.measurements.label()` to separate the recognition or multiple cars and label each detection. The bounding boxes are then calculated and overlaid on the last frame of video.


### Detection Pipeline
#### SVM Detection - Temporal Heatmap - Thresholding

![alt text][image4]


### Final Resulting Image
![alt text][image6]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One the main problems with this project was balancing execution speed/accuracy. Even with detection accuracy of 98% there were still some false positives and this is to be expected due to the number of windows and frames. I was able to solve this issue with temporal heatmaps, however fine-tunning the number of frames to store and the threshold value was tricky and required some experimentation and analysis of the debugging footage.

The pipeline will most likely fail in more harsh and varied environmental conditions such as harsh lighting or when the color of the background and the car cannot be discerned clearly. This could be improved with some data augmentation and some new diverse data image samples.

###### __AREAS FOR IMPROVEMENT__

* To further reduce false positives on the classifier side we could implement hard negative mining
* Most of the non-vehicle images were extracted from the project video, it would be best to extract from a variety of sources with varying conditions to further help generalize the classifier
* Currently the feature vector is pretty large, perhaps an implementation using decision trees could be explored
* We could calculate centroid of each label and predict movement/direction of the car and do a targeted search in that region with a more dense grid.
* Lastly we could use SSD and R-CNN approaches for image segmentation for faster and more accurate detections.
