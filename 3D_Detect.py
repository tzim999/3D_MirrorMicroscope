'''
Tracks objects and their reflections from video (mirror microscope)
Movement detector using Median Filter (subtrack median calculated from random 
frames before detection to remove static background)

V6 10.18.23 Release notes

Thomas Zimmerman, IBM Research-Almaden, Center for Cellular Construction
This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297, 
Center for Cellular Construction. Disclaimer:  Any opinions, findings and conclusions 
or recommendations expressed in this material are those of the authors and do not 
necessarily reflect the views of the National Science Foundation.
'''

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

##################### USER SET VARIABLES ############################
vid=r'StentorMirrorMicroscope.mov'          # (input) Video to be analyzed
dataFileName='3D_Detect_FullStentor_3.csv'  # (output) detection output file 

##################### PROGRAM VARIABLES ############################
xRez=640; yRez=480;     # video is resized to this rez
xRez=960; yRez=540;     # video is resized to this rez
PROCESS_REZ=(xRez,yRez)
NUMBER_MEDIAN_FRAMES=25  
PLOT_UPDATE_FRAME=300       # plot every N frames
DELAY=10 # waitKey between frames

MAX_OBJ=200; MAX_COL_OBJ=14
obj=np.zeros((MAX_OBJ,MAX_COL_OBJ))
FRAME,ID,XC,YC,ZC,XR,YR,RADIUS_C,RADIUS_R,SLOPE_C,SLOPE_R,SLOPE_MUTUAL,PAIRED,SAVE=range(MAX_COL_OBJ) #XC,YC,ZC=real object XR,YR=reflection
header='FRAME,ID,XC,YC,ZC,XR,YR,RADIUS_C,RADIUS_R,SLOPE_C,SLOPE_R,SLOPE_MUTUAL,PAIRED,SAVE'
data=np.zeros((0,MAX_COL_OBJ))
keyState=0;
keyBuf=np.zeros((256),dtype=int)

keyBuf[ord('t')]=27     # 27 threshold after blur
keyBuf[ord('a')]=32     # 19 min area an object must have
keyBuf[ord('A')]=531    # 641 max area
keyBuf[ord('x')]=563    # int(xRez/2) # x center
keyBuf[ord('y')]=209    # int(yRez/2) # y center
keyBuf[ord('d')]=120    # max pixels between obj and reflection
keyBuf[ord('w')]=32     # max width height

keyBuf[ord('z')]=120    # zmax     # int(yRez/2) # y center
keyBuf[ord('s')]=30     # 20 maxDMR, how close slope between obj and reflection is to center vector (deg)
keyBuf[ord('b')]=1      # blur size

##################### FUNCTIONS ############################
clip = lambda x, l, u: l if x < l else u if x > u else x # clip routine clip(var,min,max)

def plotXYZ(obj,frameCount,xList,yList,zList):
    for i in range(len(obj)):
        if obj[i,SAVE]==1:
            xList.append(int(obj[i,XC]))  # must make i image (not reflection, i.e. further radial distance)
            yList.append(int(obj[i,YC]))
            zList.append(int(obj[i,ZC]))
        
    if frameCount%PLOT_UPDATE_FRAME==0 and frameCount!=0:
        ax = plt.axes(projection='3d')   
        ax.axes.set_xlim3d(left=0, right=xRez) 
        ax.axes.set_ylim3d(bottom=0, top=yRez) 
        ax.axes.set_zlim3d(bottom=0, top=zMax) 
        ax.scatter3D(xList,yList,zList,marker='.')
        plt.show()         
        xList=[]; yList=[]; zList=[]
    return(xList,yList,zList)

def slopeDegRadius(xx,yy,xc,yc):
    # returns angle in degrees (0 to 360) and radius (xc,yc to optical center)
    dx=xx-xc; dy=yy-yc; 
    r=math.sqrt(dx*dx + dy*dy)
    if dx==0:
        dx=0.001
    s=dy/dx
    ss=int(math.degrees(math.atan(s)))
    if dx<0:
        ss+=180
    elif dx>0 and dy<0:
        ss+=360
    return(ss,r)


def getMedian(VID, numberMedianFrames, PROCESS_REZ):
    AGC_SETTLE=1  # give video image autobrightness (AGC) time to settle

    # Open Video
    print ('openVideo:', VID)
    cap = cv2.VideoCapture(VID)
    if(cap.isOpened()):  # get median if video can be opened
        maxFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('maxFrame', maxFrame)
        # Randomly select N frames
        print('calculating median with',numberMedianFrames,'frames')
        frameIds = AGC_SETTLE + (maxFrame - AGC_SETTLE) * np.random.uniform(size = numberMedianFrames)
        frames = [] # Store selected frames in an array
        for fid in frameIds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            colorIM = cv2.resize(frame, PROCESS_REZ)
            grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)
            frames.append(grayIM)  
        medianFrame = np.median(frames, axis = 0).astype(dtype = np.uint8)     # Calculate the median along the time axis   
        cap.release()
    else:
        print('ERROR: Could not open video', VID)
        medianFrame=0
    return (medianFrame)

def processKey(key):
    global keyState,blurK,thresh,minArea,maxArea,maxW,xCenter,yCenter,zMax,maxDistance,maxDMR;

    if key==ord('='):
        keyBuf[keyState]+=1
    elif key==ord('+'):
        keyBuf[keyState]+=10
    elif key==ord('-') and keyBuf[keyState]>0:
        keyBuf[keyState]-=1
    elif key==ord('_') and keyBuf[keyState]>10:
        keyBuf[keyState]-=10
    else:
        keyState=key
        print(chr(key),keyBuf[keyState])
    print(keyBuf[keyState])
    
    blurK=keyBuf[ord('b')]*2+1 # must be an odd value
    thresh=keyBuf[ord('t')]
    minArea=keyBuf[ord('a')]
    maxArea=keyBuf[ord('A')]
    maxW=keyBuf[ord('w')]
    xCenter=keyBuf[ord('x')]
    yCenter=keyBuf[ord('y')]
    zMax=keyBuf[ord('z')]
    maxDistance=keyBuf[ord('d')] # max pixels between obj and reflection
    maxDMR=keyBuf[ord('s')] # max delta slope between objects and object to optical center
    return(blurK,thresh,minArea,maxArea,maxW,xCenter,yCenter,zMax,maxDistance,maxDMR)

def alligned(obj,i,j): # return 1 if both objects radius vector lined up to center
    slopeClose=0        
    slopeI=obj[i,SLOPE_C]
    xi=obj[i,XC]; yi=obj[i,YC]; xj=obj[j,XC]; yj=obj[j,YC]; 
    (slopeIJ,length)=slopeDegRadius(xi,yi,xj,yj)
    if slopeIJ>270:
        slopeIJ-=360
    if slopeI>270:
        slopeI-=360
    dSlope=abs(slopeI-slopeIJ)
    
    if dSlope<maxDMR:
        slopeClose=1
    return(slopeClose)
    
def allignedX(obj,i,j): # return 1 if both objects radius vector lined up to center
    align=0     
    xci=obj[i,XC]; yci=obj[i,YC]; 
    xcj=obj[j,XC]; ycj=obj[j,YC]; 
    (Mij,r)=slopeDegRadius(xci,yci,xcj,ycj)  # slope of line between paired objects
    Mi=obj[i,SLOPE_C]
    (Mij,r)=slopeDegRadius(xci,yci,xcj,ycj)  # slope of line between paired objects
    DMR=abs( Mi%90 - Mij%90 )     # slope of pair to slope of one object 
    if DMR<maxDMR: 
        align=1
    return(align)

def calcZ(obj,i,j):
    # simple difference, should take r into consideration!
    dx = obj[i,XC] - obj[j,XC]
    dy = obj[i,YC] - obj[j,YC]; 
    d=math.sqrt(dx*dx + dy*dy)
    return(d)    
               
def drawVector(obj,i,colorIM):  
    # draw line from objects to optical center. Place circle around real object
    radius=5; thickness=2; 
    xcC=obj[i,XC]; ycC=obj[i,YC]; 
    xcR=obj[i,XR]; ycR=obj[i,YR]; 
    cv2.line(colorIM,(xcC,ycC),(xcR,ycR),(255,0,0),4)
    cv2.line(colorIM,(xcC,ycC),(xCenter,yCenter),(128,128,128),1)
    cv2.line(colorIM,(xcR,ycR),(xCenter,yCenter),(128,128,128),1)    
    cv2.circle(colorIM, (xcC,ycC), radius, (0,0,255), thickness)
    return(colorIM)

def detect(binaryIM):
    # find objects in binary image
    contourList, hierarchy = cv2.findContours(binaryIM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # all countour points, uses more memory
    objIndex=0
    obj=np.zeros((MAX_OBJ,MAX_COL_OBJ),dtype='int')
    for objContour in contourList:                  # process all objects in the contourList
        area = int(cv2.contourArea(objContour))     # find obj area     
        PO = cv2.boundingRect(objContour)
        x0=PO[0]; y0=PO[1]; x1=x0+PO[2]; y1=y0+PO[3]
        w=x1-x0; h=y1-y0; xc=int(x0+(w/2)); yc=int(y0+(h/2))
        if area>minArea and area<maxArea and objIndex<MAX_OBJ:
            cv2.rectangle(binaryIM, (x0,y0), (x1,y1),255, 1) # place read rectangle around object with good area
            if w<maxW and h<maxW and objIndex<MAX_OBJ:                           # only detect large objects       
                cv2.rectangle(colorIM, (x0,y0), (x1,y1), (0,255,0), 1) # place GREEN rectangle around each object with good area, width and height
                (slope,r)=slopeDegRadius(xc,yc,xCenter,yCenter) # slope is line from object to optical center
                obj[objIndex,RADIUS_C]=r  
                obj[objIndex,SLOPE_C]=slope  
                obj[objIndex,XC]=xc
                obj[objIndex,YC]=yc
                objIndex+=1
        if objIndex>=MAX_OBJ:
            print("MAX OBJECTS EXCEEDED!")
    objCount=objIndex    
    return(obj,objCount)

def getBinaryImage(colorIM,medianFrame):
    grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)
    diffIM = cv2.absdiff(grayIM, medianFrame)   # Calculate absolute difference of current frame and the median frame
    blurIM = cv2.GaussianBlur(diffIM, (blurK, blurK), 0)
    ret, binaryIM = cv2.threshold(blurIM, thresh, 255, cv2.THRESH_BINARY) # threshold image to make pixels 0 or 255 
    cv2.circle(colorIM, (xCenter,yCenter), 5, (255,255,0), 1)
    cv2.circle(colorIM, (xCenter,yCenter), int(maxDistance/2), (255,255,0), 1)
    return(colorIM,binaryIM)

def checkDistance(obj,i,j): # check distance between points
    distanceOK=0
    xci=obj[i,XC]; yci=obj[i,YC]; 
    xcj=obj[j,XC]; ycj=obj[j,YC]; 
    dx=abs(xci-xcj); dy=abs(yci-ycj); 
    d=math.sqrt(dx*dx+dy*dy)
    if d<maxDistance:
        distanceOK=1
    return(distanceOK)

def pairObjects(obj,objCount,frameCount,colorIM):
    if objCount==0:         # if no objects, just return obj array
        return(obj,colorIM) 
    #sort objects by distance 
    do=objCount             # short hand
    do2=do*do               # short hand
    d=np.zeros((do2))       # array holding distance between all combinations objects
    for i in range(objCount):        # search all objecs
       for j in range(objCount):     # search all possible pairs
           if i!=j:                             # do not pair to self
               xci=obj[i,XC]; yci=obj[i,YC]; 
               xcj=obj[j,XC]; ycj=obj[j,YC]; 
               dx=abs(xcj-xci); dy=abs(ycj-yci); 
               ij=(i*do+j)      # creating a 1D index from i and j to enable sorting later
               d[ij]=math.sqrt(dx*dx+dy*dy)
    di=np.argsort(d)    # get index of distances sorted in ascending order (small to large)
    
    # find pairs with smallest distance 
    obj[:,PAIRED]=0 # start with all objects not paired
    obj[:,SAVE]=0   # start with all objects not save worthy
    for ij in range(do2):
        k=di[ij] # index of low distance pair
        if k!=0: # zero distance impossible so ignore
            i=int(k/objCount); j=k-i*objCount; 
            if i!=j and obj[i,PAIRED]==0 and obj[j,PAIRED]==0 and alligned(obj,i,j) and checkDistance(obj,i,j): 
                obj[i,PAIRED]=1
                obj[j,PAIRED]=1
                # force i to be real object (larger radius) and j to be reflection
                if obj[i,RADIUS_C]<obj[i,RADIUS_C]:
                    temp=obj[i,XC];         obj[i,XC]=obj[j,XC];                obj[j,XC]=temp
                    temp=obj[i,YC];         obj[i,YC]=obj[j,YC];                obj[j,YC]=temp
                    temp=obj[i,RADIUS_C];   obj[i,RADIUS_C]=obj[j,RADIUS_C];    obj[j,RADIUS_C]=temp
                    temp=obj[i,SLOPE_C];    obj[i,SLOPE_C]=obj[j,SLOPE_C];      obj[j,SLOPE_C]=temp
                obj[i,XR]=obj[j,XC] # XR,YR is center of reflection
                obj[i,YR]=obj[j,YC]
                obj[i,RADIUS_R]=obj[j,RADIUS_C]
                obj[i,SLOPE_R]=obj[j,SLOPE_C]
                obj[i,FRAME]=frameCount
                obj[i,SAVE]=1 # save pair at end of program
                obj[i,ZC]=calcZ(obj,i,j)
                colorIM=drawVector(obj,i,colorIM)      
    return(obj,colorIM)

def saveData(obj,data):
    obj[:,ID]=-1 # force all ID = -1 to indicate none assigned since detection does not track 
    (r,c)=obj.shape
    for i in range(len(obj)):
        if obj[i,SAVE]==1:
            d=np.reshape(obj[i],(1,c))
            data=np.append(data,d,axis=0)           
    return(data)

##################### MAIN ############################
fig = plt.figure()
ax = plt.axes(projection='3d')
(blurK,thresh,minArea,maxArea,maxW,xCenter,yCenter,zMax,maxDistance,maxDMR)=processKey(ord('s'))
medianFrame = getMedian(vid, NUMBER_MEDIAN_FRAMES, PROCESS_REZ) # create median frame

cap = cv2.VideoCapture(vid)   
xFullRez=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
yFullRez=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print ('Original Rez:',xFullRez,yFullRez)
print ('totalFrames:',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

# start processing video 
frameCount=0
xList=[]; yList=[]; zList=[] # for xyz plotting
while(cap.isOpened()):

    # capturing video frame 
    ret, frameFull = cap.read()
    if not ret:
        print ('Done reading video')
        break
    colorIM = cv2.resize(frameFull, (xRez, yRez)) # resize for faster processing
      
    # process keyboard
    key=cv2.waitKey(DELAY) & 0xFF
    if key == ord('q'):
        break
    if key!=255:
        (blurK,thresh,minArea,maxArea,maxW,xCenter,yCenter,zMax,maxDistance,maxDMR)=processKey(key)

    (colorIM,binaryIM) = getBinaryImage(colorIM,medianFrame)    # process image and detect objects
    (obj,objCount)=detect(binaryIM)            
    (obj,colorIM)=pairObjects(obj,objCount,frameCount,colorIM) # find pair closest to each other for all combinations of pairs
    (xList,yList,zList)=plotXYZ(obj,frameCount,xList,yList,zList)  
    data=saveData(obj,data)
    
    cv2.imshow('binaryIM',binaryIM)
    cv2.imshow('colorIM',colorIM)
    frameCount+=1      

# quit program 
np.savetxt(dataFileName,data,header=header,delimiter=',',fmt='%i')  
print ('done')
cap.release()
cv2.destroyAllWindows()
