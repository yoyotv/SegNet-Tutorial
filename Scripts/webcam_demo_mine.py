import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time
import os


#sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:

caffe_root = '/SegNet/caffe-segnet/'
sys.path.insert(0, caffe_root + 'python')

import caffe


# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
args = parser.parse_args()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

caffe.set_mode_gpu()

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(args.colours).astype(np.uint8)

cv2.namedWindow("Input")
cv2.namedWindow("SegNet")
cv2.namedWindow("Merge")

#cap = cv2.VideoCapture(0) # Change this to your webcam ID, or file name for your video file
cap = cv2.VideoCapture('demo_video.mp4') # Change this to your webcam ID, or file name for your video file
#cap = cv2.VideoCapture('1.jpg')
#cap=cv2.imread('1.jpg')
rval = True

while rval:
	start = time.time()
	rval, frame = cap.read()
        #frame=cap
        if rval == False:
            break

	end = time.time()
	print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

	start = time.time()
	frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
	input_image = frame.transpose((2,0,1))
	# input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
	input_image = np.asarray([input_image])
	end = time.time()
	print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

	start = time.time()
	out = net.forward_all(data=input_image)
	end = time.time()
	print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

	start = time.time()
	segmentation_ind = np.squeeze(net.blobs['argmax'].data)
	segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

	cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
	segmentation_rgb = segmentation_rgb.astype(float)/255

	end = time.time()
	print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'




        #This part is edge detection by cv2#####
        img1=cv2.GaussianBlur(frame,(5,5),0)   #
                                               #
        canny=cv2.Canny(img1,50,10)            #
                                               #
        merge=segmentation_rgb                 #
        ########################################




	cv2.imshow("Input", frame)

	cv2.imshow("SegNet", segmentation_rgb)
        """
        for i in range(360):
          for j in range(480):
            if canny[i,j]==255:
              merge[i,j,:]=255
        """
	cv2.imshow("Merge", canny)




        #raw_input()
	#cv2.imwrite('temporary/SegNet.jpg',frame)
        #print(segmentation_rgb.shape)
        #np.save('temporary/SegNet.npy',segmentation_rgb)
        #np.savetxt('temporary/SegNet.txt',segmentation_rgb)




	key = cv2.waitKey(1)
	if key == 27: # exit on ESC
	    break
        
        
cap.release()
cv2.destroyAllWindows()

