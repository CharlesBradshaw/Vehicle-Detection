# Vehicle Detection

<a href="https://youtu.be/3MRTCJjzzek" target="_blank"><img src="http://img.youtube.com/vi/3MRTCJjzzek/0.jpg" 
alt="Video Link" width="1280" height="720" border="10" /></a>

## Project Goals

*Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
*Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
*Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
*Create a heat map of recurring detections frame by frame to reject outliers
*Estimate a bounding box for vehicles detected.
*Run the pipeline on a video stream and  and follow detected vehicles.


[image1]: /markdown_images/hog.png "Histogram of Gradients"
[image2]: /markdown_images/hoc.png "Histogram of Colors"
[image3]: /markdown_images/bin.png "Spatial Binning"
[image4]: /markdown_images/search.png "Restricted Search"
[image5]: /markdown_images/output.png "Output"




## Image Processing Pipeline



### 1. Feature: Histogram of Gradients
### 2. Feature: Histogram of Colors
### 3. Feature: Spatial Binning
### 4. Train on SVM (Business Requirement)
### 5. Search for Optimal Color Space
### 6. Search for Optimal Spatial Binning Size
### 7. Scale Size Restricted Search
### 8. Scan Image Using Different Sizes
### 9. Use Heatmap Over Frames to Isolate Vehicles
### 10. Draw Boxes Over Vehicles


## 1. Feature: Histogram of Gradients

One way to gain information about the presence or lack of a vehicle in an image would be to look for the shape of a vehicle within an image. Previously this has been done using a Histogram of Gradients (HOG). The idea is to split the image up into small cells, and calculate the gradient of each cell, then pass the localized gradients to a classifier. HOGs have been used to detect pedestrians, so it makes sense to apply it to vehicle detection.
The histogram is created with 9 "bins" for all 360 degrees. This allows for a model using a HOG to have a high tolerance for variance when it comes to the shape of the car. That is important as it is means the model is already more likely to generalize and not overfit.
In this project I will be passing the HOG feature to an SVM. This approach is different from a convolutional neural net. Since there is no version of "shared weights" in an SVM, the network will not be location invariant. In other words if this network was trained on cars apperaing in the top left of an image, it would not necessarily be able to recognize a car in the bottom right. This problem is "solved" by having the vehicles nearly centered in the training and test data.


!["HOG"][image1]

### 2. Feature: Histogram of Colors

Another way to gain information from an image is with a Histogram of Colors. The idea is that while it's possible a car will be camouflaged to have the same color tally as distant hill, it's unlikely and therefore we can gain information by assuming it isnt. This feature certainly isn't enough to detect a car on it's own, but it absolutely can give information.

!["HOC"][image2]

### 3. Feature: Spatial Binning

The last idea is to just give the model the image. Although a full resolution image would have too much data for the model to learn, therefore making it harder for the model to generalize, a scaled down image could certainly help. As seen below it looks like 32x32 doesn't lose much data, 16x16 is enough but it starts to get blurry, and 8x8 isn't enough

!["BIN"][image3]

### 4. Train on SVM (Business Requirement)

As it turns out, many of Udacity's partners use older machine learning algorithms in their self driving car pipelines. My understanding is that my goal should be to get experience with SVM's and more importantly understand their limitations. From here I normalize and unroll all of the features.

### 5. Search for Optimal Color Space

Earlier I plotted some images in 3D to get a grasp of how different color spaces can give more information when trying to detect a vehicle. After plotting the images I had a rough idea that RGB wasn't the best color space, but not much beyond a gut feeling about which ones would work. Below I tested the different color spaces on the entire dataset to get a more quantitative understanding of what color space works the best. HSV and HLS preformed the best with RGB coming in last place.

### 6. Search for Optimal Spatial Binning Size

Earlier I made the assumption that using a spatial size of (32,32) didn't lose too much data, and while (16,16) was lossy, it still had enough data to help make decisions. Below I tested equally spaced spatial sizes. Note that I am not doing a grid search of color spaces and spatial sizes because I don't believe they are heavily correlated. Although, it is possible that one of the color spaces works much when downsized. That said, I found that for HSV the resolution of the image didn't have much effect on the output. This implies that the actual image isn't useful, or isn't as useful as the other features.
That said 64x64 did give my model a bit of an edge over the 8x8 version. I'm worried that including it could cause overfitting, but I'll have to wait until I'm working with the video to see if it hurts more than it helps.

### 7. Scale Size Restricted Search

Vehicles get smaller the farther away they are from the camera. On a flat plane, aka a road, cars will get smaller the closer they get to the horizon, so it doesn't make sense to search at the bottom of the image for the smallest cars. Using this logic, I can search from about the middle to about the bottom of the image for the largest cars, then incrementally step down to the smallest size near the center of the image. This can be accomplished by doing a controlled cropping of the image.

!["Restricted Search"][image4]
### 8. Scan Image Using Different Sizes

The last section implied using different image search sizes, and gave an idea of how to restrict the search space. The logical next step is to implement multiple sizes for scanning

### 9. Use Heatmap Over Frames to Isolate Vehicles

The best accuracy I was able to get was around 96%. My scanning pipeline runs on a few hundred images, so I should expect a significant number of false positive images. Obviously detecting a "phantom car" could cause serious harm on a highway, especially if the reaction is slamming on the breaks with a car close behind. One way to get around this is to use a Heatmap over multiple images. In theory, if my model is doing it's job correctly, then over multiple frames and multiple sizes it should detect a true car multiple times. Therefore if I look at a Heatmap of the cars and isolate the "hot points" I should have a more reliable way to detect cars.

### 10. Draw Boxes Over Vehicles

The last step is to give a visual representation of the detected boxes.
!["Output"][image5]

### 11. Lessons Learned

As can be seen in the video, my model is far from perfect, but also I've learned some of the pitfalls of older machine learning models. I'm tempted to augment the data, or use more data for training the SVM, but SVM's run with a time complexity of O(n^3), so adding more data is not a means to an end. For example, if I were to augment data, it would be data of sides of cars as my model appears to have difficulty reading the side of a car in comparison to the bottom of the car.

Another way my model could be improved is to detect the vehicles and take an average over multiple frames. While it may seem like I am doing this with the Heatmap, the difference is that an averaged output would be far more user friendly, and it would be more likely to represent the entire car.

