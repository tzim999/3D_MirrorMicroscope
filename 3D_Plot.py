# -*- coding: utf-8 -*-
"""
Plots 3D trajectory of each object (ID) using a red heat scale to represent time (from mirror microscope)
Limit how many points are in each plot (minPoints to maxPoints)

V4 10.18.23 Release notes

Thomas Zimmerman, IBM Research-Almaden, Center for Cellular Construction
This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297, 
Center for Cellular Construction. Disclaimer:  Any opinions, findings and conclusions 
or recommendations expressed in this material are those of the authors and do not 
necessarily reflect the views of the National Science Foundation.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

##################### USER SET VARIABLES ############################
trackFile=r'3D_Track.csv'   # (input) location of track file
plotDir=r'\\3D_Plots\\'     # (output) where to save plots, located in working directory

wd=os.getcwd()
plotDir=wd+plotDir  # put plotDir into working directory

##################### PROGRAM VARIABLES ############################
MAX_OBJ=20; MAX_COL_OBJ=14
FRAME,ID,XC,YC,ZC,XR,YR,RADIUS_C,RADIUS_R,SLOPE_C,SLOPE_R,SLOPE_MUTUAL,PAIRED,SAVE=range(MAX_COL_OBJ) #XC,YC,ZC=real object XR,YR=reflection

##################### FUNCTIONS ############################
def plotXYZ(selectID,xList,yList,zList,label,frame):
    totalPoints=len(xList)
    maxPoints=500;  minPoints=100
    size=5
    sections=int(totalPoints/maxPoints) # break into several plots
    for s in range(sections):
        a=s*maxPoints; b=a+maxPoints    # compute range of points to plot
        if (b-a)>minPoints:
            t=range(a,b)                    # time will be encoded in color shade
            ax = plt.axes(projection='3d')
            #ax.set_facecolor("black")
            p=ax.scatter(xList[a:b],yList[a:b],zList[a:b],c=t,cmap=plt.cm.get_cmap("Reds"),s=size)
            plt.colorbar(p)
            fileName='ID='+str(selectID)+ '   Frames='+ str(a) +'-'+ str(b)
            plt.title(fileName)
            plt.savefig(plotDir+fileName+'.png')
            plt.show()  
    return         

#################### MAIN ###############################
data=np.loadtxt(trackFile,delimiter=',',skiprows=1,dtype='int')
maxID=max(data[:,ID])

for selectID in range(maxID):
    xList=[]; yList=[]; zList=[]; label=[]; frame=[] # for xyz plotting
    for i in range(len(data)):
        if data[i,ID]==selectID:
            xList.append(data[i,XC])
            yList.append(data[i,YC])
            zList.append(data[i,ZC])
            label.append(data[i,ID])
            frame.append(data[i,FRAME])
    plotXYZ(selectID,xList,yList,zList,label,frame)

    