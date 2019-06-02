# Augmented Reality Tag Detection and Tracking


## **PROJECT DESCRIPTION**

The aim of this project is to detect custom AR Tags 
- detection in itself is a three-step process of encoding, detection and tracking.
- Then an image (Lena) will be superimposed onto the tag. 
- Finally, a virtual 3D cube will be placed on the tag

### Encoding Stage

<p align="center">
  <img src="/Images/ref_marker_grid.png" alt="Reference Marker">
</p>

 
- The tag can be decomposed into an 8 × 8 grid of squares, which includes a padding of 2 squares width
along the borders. This allows easy detection of the tag when placed on white background.

- The inner 4 × 4 grid (i.e. after removing the padding) has the orientation depicted by a white square in
the lower-right corner. This represents the upright position of the tag.

- Lastly, the inner-most 2 × 2 grid (i.e. after removing the padding and the orientation grids) encodes the
binary representation of the tag’s ID, which is ordered in the clockwise direction from least significant bit
to most significant. So, the top-left square is the least significant bit, and the bottom-left square is the
most significant bit

### Detection Stage

<p align="center">
  <img src="/Images/Detection.png" alt="Detect Tag" width="200"/>
</p>

The detection stage involves finding the AR Tag from a given image 

### Tracking Stage

<p align="center">
  <img src="/Images/track.png" alt="Track Tag">
</p>

The tracking stage will involve keeping the tag in view throughout the sequence and performing image processing operations based on the tag’s orientation and position. This is done using Homography. Refer to the Document [Supplementary Homography](https://github.com/adheeshc/Augmented-Reality-Tag-Detection-and-Tracking/blob/master/Report/Supplementary_Homography.pdf)
in the Report folder for more details.

### Superimposing Lena

<p align="center">
  <img src="/Images/Lena_on_Tag.gif" alt="Lena on Tag">
</p>

The first step is to compute the homography between the corners of the template and the four corners of the tag.
Then I transform the template image onto the tag, such that the tag is “replaced” with the template such that the orientation of the transformed template image matches that of the tag at any given frame.

### Placing a Virtual 3D Cube

<p align="center">
  <img src="/Images/Cube_on_Tag.gif" alt="Cube on Tag">
</p>

The “cube” is a simple structure made up of 8 points and lines joining them. 
The first step is to compute the homography between the world coordinates (the reference AR tag) and the image plane (the tag in the image sequence). 
Then the projection matrix is calculated from the camera calibration matrix and the homography matrix and the cube is placed onto the tag


## **FILE DESCRIPTION**

- Code Folder/[Final.py](https://github.com/adheeshc/Augmented-Reality-Tag-Detection-and-Tracking/blob/master/Code/Final.py)- The final code file to do the same. (Please comment and uncomment cv2.imshow as reqd)

- Datasets folder - Contains 4 video input files 

- Images folder - Contains images for github use (can be ignored)

- Output folder - Contains output videos

- Report folder - Contains Project Report and Supplementary Homography document

## **RUN INSTRUCTIONS**

- Make sure all dependancies are met
- Ensure the location of the input video files are correct in Final.py
- Comment/Uncomment as reqd
- RUN Final.py
