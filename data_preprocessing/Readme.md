# Data Preprocessing

Given an image the function/file does the following
1. For the predefined set of classes of image to be operated on
    1. Read the image; Resize the image
    2. Draw bounding boxes
    3. Add additional data preprocessing;  Resizing; Image augmentation methods etc
    4. Save the label 

Image dimension here is 640 x 640; The intention is to allow the model to detect on different lighting conditions and different Image augmentations
    
Following set of augmentation are performed for the same
1. Horizontal filipping
    Rotates the image by a scale of 180 degree
        The new coordinates are now
        x: - DIM/2 + (DIM/2 - x_original)
        y: y
        h: h
        w: w
2. Vertical fillipping: This causes image to mirror with respect to x axis
    x: x
    y: DIM/2 + (DIM/2 - y_original)
    h: h
    w: w
3. Scaling: Positive affine Scaling
4. Negative affine scaling: done on 2 levels 
    (reason being negative scaling does not make the edges to disappear)
    x: DIM_x/2 + scale * (DIM_x/2 - old_DIM_x)
    y: DIM_y/2 + scale * (DIM_y/2 - old_DIM_y)
    h: scale * h_old
    w: scale * w_old
5. Gaussian Blurring
6. Contrastive Normalization
7. AdditiveGaussianNoise
8. Multiply

The function creates labelling window and stores the data in train/validation based on distribution probability.