# Description: https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/83ec35ee-1e02-48a5-bdb7-d244bd47c2dc/lessons/8c82408b-a217-4d09-b81d-1bda4c6380ef/concepts/b180303f-be20-4b38-a3d3-aa0ffe8d3ea0

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# print np.__path__
# Read in the image and print out some stats
image = mpimg.imread('test.jpg')
print('This image is: ',type(image), 
         'with dimensions:', image.shape)

# ('This image is: ', <type 'numpy.ndarray'>, 'with dimensions:', (540, 960, 3)) (each point has RGB value)
# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make a copy rather than simply using "="
color_select = np.copy(image)
region_select = np.copy(image)
line_image = np.copy(image)




# Define our color selection criteria
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz
red_threshold = 200 
green_threshold = 200 
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
# print(thresholds)

# [[False False False ..., False False False]
#  [False False False ..., False False False]
#  [False False False ..., False False False]
#  ..., 
#  [False False False ..., False False False]
#  [False False False ..., False False False]
#  [False False False ..., False False False]]

# What's the meaning of this
# Turn the value true into certain value
color_select[thresholds] = [0,0,0]
# print(color_select[thresholds])
# print(color_select)



# Display the image                 
plt.imshow(color_select)
# save figure before you call show, https://stackoverflow.com/questions/9012487/matplotlib-pyplot-savefig-outputs-blank-image
plt.savefig('quiz-1-color-selection.jpg')
plt.show()