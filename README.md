# Augmented Reality Tag Detection and Tracking


## **PROJECT DESCRIPTION**

The aim of this project is to detect custom AR Tags which is a three-step process of encoding, detection and tracking.
Then an image will be superimposed onto the tag. Finally, a virtual 3D cube will be placed on the tag

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
<p align="centre">
  <img src="/Images/Detection.png" alt="Detect Tag" width="200"/>
</p>
The detection stage involves finding the AR Tag from a given image 

### Tracking Stage

![Track Tag](/Images/Tracking.gif)

The tracking stage will involve keeping the tag in view throughout the sequence and performing image processing operations based on the tag’s orientation and position

## Superimposing Lena

![Lena Tag](/Images/Lena_on_tag.gif)

## Placing a Virtual 3D Cube

![Cube Tag](/Images/Cube_on_tag.gif)



#### **FILE DESCRIPTION**

Final.py - 

#### **RUN INSTRUCTIONS**
