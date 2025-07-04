# intrinsic calibration camera

Checkerboard images can be used to calibrate the camera. The calibration process estimates the intrinsic parameters of the camera, such as focal length, principal point, and lens distortion coefficients.

The image is below:
![checkerboard](img\checkerboard.png)


## Matlab

```
cameraCalibrator
```


install webcam for Matlab then use the live to capture images from the camera. Then you can use those photos to do the calibration. Finally, you can save the camera parameters and estimate error for the camera.

As the files show below:

cameraParams.mat
estimateError.mat


# optical maker tracking

## Video to demo the algorithm

![optical marker tracking](img\marker_tracking.gif)

