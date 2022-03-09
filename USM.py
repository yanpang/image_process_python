import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread(r'D:\YL\test.jpeg') # input an RGB image
scale_factor = 50

image_padding = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)

# print(image.shape)
# print(type(image.dtype))

[w,h,depth] = image.shape
image_out = np.zeros(image.shape,dtype=image.dtype) ## To generate a same size &dtype narray with input image, else the dtype will be default "float64"

#### Simle USM ####
for chan in range(0,depth):
    for i in range(1,w-1):
        for j in range(1,h-1):
            HighPass = (image_padding[i,j,chan] << 2) - image_padding[i-1,j,chan]-image_padding[i+1,j,chan]-image_padding[i-1,j-1,chan]-image_padding[i+1,j+1,chan] #HF filter kernel : 3x3
            
            Value = image[i-1,j-1,chan]+scale_factor*HighPass//100 # Value is "int32"
            if(Value > 255):  # Clip Value [0,255]
                Value = 255
            elif(Value < 0):
                Value = 0
        
            image_out[i-1,j-1,chan]=Value

cv2.namedWindow('img input')
cv2.imshow('img input',image)          
# cv2.namedWindow('img padding')
# cv2.imshow('img padding',image_padding)
cv2.namedWindow('img output')
cv2.imshow('img output',image_out)

#### Normal USM ####
image_blur_1  = cv2.GaussianBlur(image,(5,5),0.5) #Gaussian kernel : 5x5 ; Sigma = 0.2
image_blur_2  = cv2.GaussianBlur(image,(5,5),1.0) #Gaussian kernel : 5x5 ; Sigma = 0.5 
image_blur_3  = cv2.GaussianBlur(image,(5,5),2.2) #Gaussian kernel : 5x5 ; Sigma = 1.0 
image_usm_1 = cv2.addWeighted(image, 1.5, image_blur_1, -0.5, 0) ## gamma(1.5*image - 0.5*image_blur)
image_usm_2 = cv2.addWeighted(image, 1.5, image_blur_2, -0.5, 0) 
image_usm_2 = cv2.addWeighted(image, 1.5, image_blur_3, -0.5, 0)

cv2.namedWindow('img blur sigma=0.5')
cv2.imshow('img blur sigma=0.5',image_blur_1)    
cv2.namedWindow('img usm sigma=0.5')
cv2.imshow('img usm sigma=0.5',image_usm_1)  

cv2.namedWindow('img blur sigma=1.0')
cv2.imshow('img blur sigma=1.0',image_blur_2)    
cv2.namedWindow('img usm sigma=1.0')
cv2.imshow('img usm sigma=1.0',image_usm_2)  

cv2.namedWindow('img blur sigma=2.2')
cv2.imshow('img blur sigma=2.2',image_blur_3)   
cv2.namedWindow('img usm sigma=2.2')
cv2.imshow('img usm sigma=2.2',image_blur_3)   

##compared the different Sigma value result, it shows higher Sigma -> more blur -> sharpen unknown

cv2.waitKey()
cv2.destroyAllWindows()




