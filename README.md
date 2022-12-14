# Tiny-Instance-Segmentation

A ground up implementation of instance segmentation netowrk in pytorch with end goal of deploying in low powered edge device like Intel neural compute stick and rasberry pie.

# Tiny Network:
Object detection and Instance Segmentation Model


Contact: [asadismaeel@gmail.com]. Any questions, discussions or contributions are highly welcomed! 

## Abstract 

Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this work, we take a different approach we take an approach which is similar to center net. We model an object as a single point -- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as xmin, xmax, ymin and ymax. Different from center net we regress 4 points instead of 2 sizes of height and width.
Our Model also makes the training data processing simpler by having Dirac Delta (point) distribution instead of Gaussian distribution for each object. Our center point based approach, Tiny Net, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors.
Purpose of Tiny net is not just another research project only but also used in real world projects in production.

## Highlights

- **Simple:** One-sentence method summary: use keypoint detection technic to detect the bounding box center point and regress to all other object properties like bounding box size, 3d information, and pose.

- **Versatile:** The same framework works for object detection, 3d bounding box estimation, and multi-person pose estimation with minor modification.

- **Fast:** The whole process in a single network feedforward. No anchor boxes and NMS post processing is needed

- **Strong**: Our best single model achieves *45.1*AP on COCO test-dev.

- **Easy to use:** We provide user friendly testing API and webcam demos.

## Main results


