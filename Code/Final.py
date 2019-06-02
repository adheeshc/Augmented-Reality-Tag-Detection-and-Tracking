import cv2
import numpy as np
import math 
import copy

np.set_printoptions(suppress = True)

# Create a VideoCaptureaq object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('multipleTags.mp4')
lena = cv2.imread('Lena.png')
ref = cv2.imread('ref_marker.png')
lenag = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)




#Rotation function for Lena
def rotatenew(l, n):
    return l[n:] + l[:n]

def rotate(index,):
    if index !=0:
        if index == 1:
            return 3
        if index == 2:
            return 1
        if index ==3:
            return 2
    else:
        return 0

def homography(m1, m2):
    count = 0
    A = np.empty((8,9))
    for i in range(0, len(m1)):
        x1 = m1[i][0]
        y1 = m1[i][1]

        x2 = m2[i][0]
        y2 = m2[i][1]

        A[count] = np.array([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A[count + 1] = np.array([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
        
        count = count + 2
    
    A1 = np.array(A[:,0:8])
    B = -1*np.array(A[:,8])
    
    H = np.linalg.inv((np.transpose(A1)).dot(A1)).dot(((np.transpose(A1)).dot(B)))
    H = np.append(H, 1)

    H = np.reshape(H, (3,3))

    return H

def orientation(tag):
    t = cv2.resize(tag,(5,5))
    new = t[1:4,1:4]
    if new[0][0] >= 200:
        return 0
    if new[0][2] >= 200:
        return 1
    if new[2][0] >= 200:
        return 2
    if new[2][2] >= 200:
        return 3


def projection_matrix(camera_parameters, homography):
    
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    R_matrix = np.dot(np.linalg.inv(camera_parameters), homography)
    col1 = R_matrix[:, 0]
    col2 = R_matrix[:, 1]
    col3 = R_matrix[:, 2]
    
    # normalise vectors
    normal = math.sqrt(np.linalg.norm(col1, 2) * np.linalg.norm(col2, 2))
    rotation1 = col1/ normal
    rotation2 = col2 / normal
    translation = col3 / normal
    
    # compute the orthonormal basis
    c = rotation1 + rotation2
    p = np.cross(rotation1, rotation2)
    d = np.cross(c, p)
    rotation1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rotation2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rotation3 = np.cross(rotation1, rotation2)
    
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rotation1, rotation2, rotation3, translation)).T
    return np.dot(camera_parameters, projection)

#Encoding Scheme
img = cv2.imread('ref_marker.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def endScheme(image):  
    img = cv2.resize(image,(8,8)) 
    return img

def scheme2x2(image):
    
    temp1=0
    temp2=0
    temp3=0
    temp4=0
    
    image2x2 = image[3:5,3:5]
    
    if image2x2[0][0]>200:
        temp1 =  8
    else:
        temp1 = 0
        
    if image2x2[0][1]>200:
        temp2 =  4
    else:
        temp2 = 0
        
    if image2x2[1][1]>200:
        temp3 =  2
    else:
        temp3 = 0
        
    if image2x2[1][0]>200:
        temp4 =  1
    else:
        temp4 = 0
    
    return temp1+temp2+temp3+temp4
 
K= np.array(np.transpose([[1406.08415449821, 0, 0],[2.20679787308599, 1417.99930662800, 0],[1014.13643417416, 566.347754321696, 1]]))

white = (255,255,255)
whiteframe = np.zeros([512,512,3],np.uint8)
whiteframe[:][:][:] = 255
NEW_XY = np.where(np.all(whiteframe == white, axis=-1))
X = NEW_XY[0]
Y = NEW_XY[1]
Z = np.ones(len(X))
coordinates=np.array([Y,X,Z])

count = 0

# Check if camera opened successfully
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  cntrs = []
  contour_coord = []
  cpts = []
  
  if ret == True:
        
#      #Countours  
        frame1 = copy.deepcopy(frame)
        frameCube=copy.deepcopy(frame)
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (thresh1,im_bw) = cv2.threshold(gray1,215,255,cv2.THRESH_BINARY)
    
        #Display the resulting frame
        image, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        corners = []
      
        for j,cnt in zip(hierarchy[0],contours):
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
            if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                if j[3] != -1:
                    corners.append(cnt)
            
  #      filled=cv2.drawContours(frame1, corners, -1, (0, 255, 0), cv2.FILLED)
        
        
        for cor in corners:
            cpts = np.float32([list(i) for i in cor])
            
                           
            #Homography calculations, see Homography function above
            tagframe = np.zeros([512,512,3])
            tagf = tagframe.astype(np.uint8)
            

            lena_pts = [[0,0],[512,0],[512,512],[0,512]] 
            lena_pts1=np.array(lena_pts)#index 0
            corr=np.float32([cpts[3],cpts[0],cpts[1],cpts[2]])   
            corr1=np.float32([cpts[0],cpts[1],cpts[2],cpts[3]])
            #Place Lena Back on Video, recalculating homography with image coordinates and lena coord.
    #        H = homography(corr,world)  
            H1 = homography(corr,lena_pts)    
            H_inv = np.linalg.inv(H1)
            
            new_coordinates = H_inv.dot(coordinates)
            new_coordinates = (new_coordinates/new_coordinates[2][:]).astype(int)
            new_coordinates = new_coordinates[:2,:]
#            
            for i in range(0,len(X)):
                tagf[X[i]][Y[i]] = frame[new_coordinates[1][i]][new_coordinates[0][i]]
#           
                
            tagf = cv2.resize(tagf,(50,50)) 
            dst = cv2.cvtColor(tagf, cv2.COLOR_BGR2GRAY)
            thresh,dst = cv2.threshold(dst,240,255,cv2.THRESH_BINARY)
            tag = endScheme(dst)
            decode = scheme2x2(tag)
            font = cv2.FONT_HERSHEY_SIMPLEX
            value= str(decode)
           # cv2.putText(frame1,value,(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
            rotations = rotate(orientation(dst))
            rotM = cv2.getRotationMatrix2D((256,256),90*rotations,1)
            result = cv2.warpAffine(lena,rotM, (512,512))
            H2_inv = np.linalg.inv(H1)
            
            new_coordinates1 = H2_inv.dot(coordinates)
            new_coordinates1 = (new_coordinates1/new_coordinates1[2][:]).astype(int)
            new_coordinates1 = new_coordinates1[:2,:]
            
            for i in range(0,len(X)):
                frame1[new_coordinates1[1][i]][new_coordinates1[0][i]] = result[X[i]][Y[i]]
            

            world1 = np.array(lena_pts1)*200 / 512
            H5=homography(world1,corr1)
            #print(H5)
            H6=H5*-1
            H5_inv=np.linalg.inv(H5)
            P =  projection_matrix(K,H5)
            cube = np.transpose(np.array([[0,0,0,1],[200,0,0,1],[200,200,0,1],[0,200,0,1],
                                [0,0,200,1],[200,0,200,1],[200,200,200,1],[0,200,200,1]]))
            imgPts = np.transpose(np.matmul(P,cube))
            for i in imgPts:
                #normalizing
                cubepts = np.int32(i[0:2]/i[2])
                #print(cubepts)
                cv2.circle(frameCube,tuple(cubepts), 5, (0,0,255), -1)
            cubeImgPts = []
            for i in imgPts:
                cubepts = np.int32(i[0:2]/i[2])
                cubeImgPts.append(tuple(cubepts))
            #base
            for i in range(0,3):
                cv2.line(frameCube,cubeImgPts[i],cubeImgPts[i+1],(0,0,255),3)
                cv2.line(frameCube,cubeImgPts[3],cubeImgPts[0],(0,0,255),3)
            #top
            for i in range(4,7):
                cv2.line(frameCube,cubeImgPts[i],cubeImgPts[i+1],(0,0,255),3)
                cv2.line(frameCube,cubeImgPts[7],cubeImgPts[4],(0,0,255),3)
            #vertical
            for i in range(0,4):
                cv2.line(frameCube,cubeImgPts[i],cubeImgPts[i+4],(0,0,255),3)
#        cv2.imwrite("TagFrame.jpg" , tagf)           
#        cv2.imwrite("FirstFrame.jpg" , frame1)  

        #frame1 = cv2.resize(frame1, (0, 0), fx=.5, fy=.5)
    ###################################
    
    # COMMENT/UNCOMMENT BELOW AS REQD

    ###################################
    
    #LENA ON TAG
        #cv2.imshow('tag', dst)
        #cv2.imshow('Video_Tag0', frameCube)
    #CUBE ON TAG
        Cube = cv2.resize(frameCube, (0,0), fx=0.7, fy=0.7)
        cv2.imshow('Cube',Cube)
    #WRITE ALL FRAMES
        cv2.imwrite("frame%d.jpg" % count, Cube)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
        count += 1
#  break    
  else: 
    break
#Release video capture object
cap.release()
cv2.destroyAllWindows()

