# -*- coding: utf-8 -*-
"""
Tracking for 3D mirror microscope detection file

V7 10.18.23 Release notes
 
Thomas Zimmerman, IBM Research-Almaden, Center for Cellular Construction
This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297, 
Center for Cellular Construction. Disclaimer:  Any opinions, findings and conclusions 
or recommendations expressed in this material are those of the authors and do not 
necessarily reflect the views of the National Science Foundation.

"""
import numpy as np
import math
import cv2

np.set_printoptions(suppress=True) # don't display numpy in scientific notation

##################### USER SET VARIABLES ############################
dataFile=r'3D_Detect.csv' # (input) Detection file to be tracked
trackFile=r'3D_Track.csv' # (output) Name of tracking file to be saved 

##################### PROGRAM VARIABLES ############################
MAX_OBJ=20; MAX_COL_OBJ=14
FRAME,ID,XC,YC,ZC,XR,YR,RADIUS_C,RADIUS_R,SLOPE_C,SLOPE_R,SLOPE_MUTUAL,OBJ_PAIRED,DISTANCE=range(MAX_COL_OBJ) #XC,YC,ZC=real object XR,YR=reflection
header='FRAME,ID,XC,YC,ZC,XR,YR,RADIUS_C,RADIUS_R,SLOPE_C,SLOPE_R,SLOPE_MUTUAL,OBJ_PAIRED,DISTANCE'
xRez=960; yRez=540;     # id display rez (based on original video resolution)
MAX_DISTANCE=500 # max acceptable distance between obj in adjacent frames
DELAY=30        # display ID delay
CAM_COL=6
CAM_X,CAM_Y,CAM_Z,CAM_ID,CAM_PAIRED,CAM_FRAME=range(CAM_COL)
cam=np.zeros((0,CAM_COL))

##################### FUNCTIONS ############################
def displayObjID(obj,oi):
    global im
    font = cv2.FONT_HERSHEY_SIMPLEX; fontScale=1; color=(0,255,255); thickness=2;
    xc=obj[oi,XC]; yc=obj[oi,YC]; objID=obj[oi,ID]; org=(xc,yc);
    im =  cv2.putText(im, str(objID), org, font, fontScale, color, thickness, cv2.LINE_AA)   
    return

def displayCamID(cam,ci):
    global im
    font = cv2.FONT_HERSHEY_SIMPLEX; fontScale=1; color=(255,0,0); thickness=2;
    xc=int(cam[ci,CAM_X]); yc=int(cam[ci,CAM_Y]); camID=int(cam[ci,CAM_ID]); org=(xc,yc);
    im =  cv2.putText(im, str(camID), org, font, fontScale, color, thickness, cv2.LINE_AA)   
    return

def findID(cam,obj,fb,fbIndex,nextID):
# find all distance combinations, sort low to high to get index, pick lowest distance for each new obj and assign ID

    (objIndex,objCount)=fb[fbIndex]
    frame=obj[objIndex,FRAME]
    p0=objIndex; p1=p0+objCount # point to begin and end of objects in frame
    # clear all paring flags
    cam[:,CAM_PAIRED]=0 # indicate that all obj in previous and current frame are not paired
    obj[p0:p1,OBJ_PAIRED]=0 # so they can be paired in the code below
    
    # match cam with closest objects
    for loop in range(len(cam)):        # go through every cam 
        bestD=99999; bestCam=-1; bestObj=-1;
        for ci in range(len(cam)): # find min cam to obj distance
            for oi in range(p0,p1):  # current Frame
                if cam[ci,CAM_PAIRED]==0 and obj[oi,OBJ_PAIRED]==0:
                    dx=cam[ci,CAM_X]-obj[oi,XC]
                    dy=cam[ci,CAM_Y]-obj[oi,YC]
                    d=math.sqrt(dx*dx + dy*dy) # 3D distance between object in last and current frame            
                    if d<bestD:     # if closer and not paired, remember this match
                        bestD=d; bestCam=ci; bestObj=oi; 
        if bestCam!=-1: # did we find a good match?
            obj[bestObj,DISTANCE]=bestD # save the distance for diagnostics
            obj[bestObj,ID]=cam[bestCam,CAM_ID] # assign current object ID (j) to object (i) in previous frame
            obj[bestObj,OBJ_PAIRED]=1  # set current object paired so it's not treated as new born
            cam[bestCam,CAM_PAIRED]=1  # set campaired so it's not assigned to multiple objects
            cam[bestCam,CAM_X]=obj[bestObj,XC] # update location of cam
            cam[bestCam,CAM_Y]=obj[bestObj,YC] # update location of cam
            cam[bestCam,CAM_Z]=obj[bestObj,ZC] # update location of cam
            cam[bestCam,CAM_FRAME]=frame # update age of cam
            #print('fbIndex',fbIndex,'bestCam',bestCam,'frame',frame)
            if frame!=obj[bestObj,FRAME]:
                print('Frame Sync error',frame,obj[bestObj,FRAME],bestObj)
            displayObjID(obj,bestObj)
            #print('match i',i,'bestJ',bestJ,'d',int(bestD),'id',int(cam[i,CAM_ID]))
 
    # display unassigned cam
    for ci in range(len(cam)):         
        if cam[ci,CAM_PAIRED]==0:
            displayCamID(cam,ci) 
            
    #create new cam and give new ID's any obj in current frame not paired (new borns, object that appear in current frame that weren't in previous frame)
    for oi in range(p0,p1):  # current Frame
        if obj[oi,OBJ_PAIRED]==0:
            obj[oi,ID]=nextID
            camX=obj[oi,XC]; camY=obj[oi,YC]; camZ=obj[oi,ZC]; 
            camID=nextID;   camPaired=0;    camAge=frame;
            v=np.array([[camX,camY,camZ,camID,camPaired,camAge]])
            cam=np.append(cam,v,axis=0)
            print('newborn frame',obj[oi,FRAME],'id',nextID,'cam shape',cam.shape)
            nextID+=1  # set up for next newborn
    return(cam,obj,nextID)


def getFrameBoundary(obj):
    # indicate the start location of a frame  
    (r,c)=obj.shape
    fb=[]   # [index to beginning of objects in frame, objects in frame]
    startFrame=obj[0,FRAME];
    startIndex=0
    objCount=0    
    for objIndex in range(r):
        frame=obj[objIndex,FRAME] 
        if frame!=startFrame:
            fb.append([startIndex,objCount])
            startIndex=objIndex
            startFrame=frame
            objCount=1
        else:
            objCount+=1
    fb.append([startIndex,objCount]) # save last obj index and obj count
    return(fb)

def initializeID(cam,obj,fb): # load objects in first frame with new ID's
    nextID=0
    (startFrame,objCount)=fb[0]
    for oi in range(objCount): # load cams with all objects of first frame 
        obj[oi,ID]=nextID
        camX=obj[oi,XC]; camY=obj[oi,YC]; camZ=obj[oi,ZC]; 
        camID=nextID;   camPaired=0;    camFrame=obj[oi,FRAME];
        v=np.array([[camX,camY,camZ,camID,camPaired,camFrame]])
        cam=np.append(cam,v,axis=0)
        nextID+=1
    return(cam,obj,nextID)

MAX_AGE=100
def ageCam(frame,objCount,cam):
    #print('frame',frame,'objects',objCount,'age',end=',')
    for ci in range(len(cam)):
        age=int(frame-cam[ci,CAM_FRAME])
        if age>MAX_AGE: # if cam hasn't been assigned for a while, move to 0,0 and update age
            print('age out',ci)
            cam[ci,CAM_X]=0; cam[ci,CAM_Y]=0; cam[ci,CAM_Z]=0; cam[ci,CAM_FRAME]=frame; 
            
###################### MAIN ##################
obj=np.loadtxt(dataFile,delimiter=',',dtype='int')
obj[:,OBJ_PAIRED]=0 # make sure all objects start as not paired, for they will be paired during tracking
fb=getFrameBoundary(obj) 

(cam,obj,nextID)=initializeID(cam,obj,fb)

for fbIndex in range(1,len(fb)):
    (oi,objCount)=fb[fbIndex]
    frame=obj[oi,FRAME]
    im=np.zeros((yRez,xRez,3),dtype='uint8') # display for showing object ID
    (cam,obj,nextID)=findID(cam,obj,fb,fbIndex,nextID)
    ageCam(frame,objCount,cam)
    cv2.imshow('obj ID',im)
    key=cv2.waitKey(DELAY)
    if key == ord('q'):
        break

np.savetxt(trackFile,obj,delimiter=',',header=header,fmt='%i')
cv2.destroyAllWindows()
print('saved tracking file',trackFile)
print('bye!')
    
