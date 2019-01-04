#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 2018

@author: cesarechavarria
"""

print('Importing retino functions...')

import cv2
import os
import glob
import numpy as np
import scipy
from scipy import misc,interpolate,stats,signal
import statsmodels.stats.multitest as mt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors
import time
import shutil


# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# MISCELANOUS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_comma_separated_args(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_theta_array(szY,szX,mode='deg'):
    #Cesar Echavarria 11/2016


    x = np.linspace(-1, 1, szX)
    y = np.linspace(-1, 1, szY)
    xv, yv = np.meshgrid(x, y)

    [radius,theta]=cart2pol(xv,yv)
    if mode=='deg':
        theta = np.rad2deg(theta)

    return theta

def array2cmap(X):
    N = X.shape[0]

    r = np.linspace(0., 1., N+1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in xrange(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in xrange(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in xrange(N)])

    rd = tuple([(r[i], rd[i], rd[i]) for i in xrange(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in xrange(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in xrange(2 * N)])


    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return colors.LinearSegmentedColormap('my_colormap', cdict, N)
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# GENERAL PURPOSE FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #


def normalize_stack(frameStack):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("No Arguments Passed!")

    frameStack=np.true_divide((frameStack-np.min(frameStack)),(np.max(frameStack)-np.min(frameStack)))
    return frameStack


def get_frame_times(planFolder):
    #Cesar Echavarria 11/2016

    # READ IN FRAME TIMES FILE
    planFile=open(planFolder+'frameTimes.txt')

    #READ HEADERS AND FIND RELEVANT COLUMNS
    headers=planFile.readline()
    headers=headers.split()

    count = 0
    while count < len(headers):
        if headers[count]=='frameCond':
            condInd=count
            break
        count = count + 1

    count = 0
    while count < len(headers):
        if headers[count]=='frameT':
            timeInd=count
            break
        count = count + 1
    planFile.close()

    # READ IN FRAME TIMES FILE ONCE AGAIN
    planFile=open(planFolder+'frameTimes.txt')
    frameTimes=[]
    frameCond=[]
    # GET DESIRED DATA
    for line in planFile:
        x=line.split()
        frameCond.append(x[condInd])
        frameTimes.append(x[timeInd])
    planFile.close()

    frameTimes.pop(0)#take out header
    frameCond.pop(0)#take out header
    frameTimes=np.array(map(float,frameTimes))
    frameCond=np.array(map(int,frameCond))
    frameCount=len(frameTimes)
    return frameTimes,frameCond,frameCount

def get_run_parameters(sourceRoot,animalID,sessID,run):
      runFolder=glob.glob('%s/%s_%s_%s_*'%(sourceRoot,animalID,sessID,run))

      frameFolder=runFolder[0]+"/frames/"
      planFolder=runFolder[0]+"/plan/"
      #GET STIM FREQUENCY
      planFile=open(planFolder+'parameters.txt')

      for line in planFile:
        if 'Cycle Rate' in line:
            break     
      idx = line.find(':')
      stimfreq = float(line[idx+1:])

      planFile.close()

      #GET FRAME RATE
      planFile=open(planFolder+'performance.txt')

      #READ HEADERS AND FIND RELEVANT COLUMNS
      headers=planFile.readline()
      headers=headers.split()

      count = 0
      while count < len(headers):
        if headers[count]=='frameRate':
            idx=count
            break
        count = count + 1

      x=planFile.readline().split()
      framerate = float(x[idx])
      planFile.close()

      return framerate, stimfreq



# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# STRUCTURAL INFO FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_surface(sourceRoot,targetRoot,animalID, sessID):

        
    #DEFINE DIRECTORY
    sourceDir=glob.glob(sourceRoot+'/'+animalID+'_'+sessID+'_surface*')
    surfDir=sourceDir[0]+'/Surface/'
    outDir=targetRoot+'/Surface/';

    if not os.path.exists(outDir):
        os.makedirs(outDir) 
        picList=glob.glob(surfDir+'*.tiff')
        nPics=len(picList)


        # READ IN FRAMES
        imFile=surfDir+'frame0.tiff'
        im0=cv2.imread(imFile,-1)
        sz=im0.shape

        allFrames=np.zeros(sz+(nPics,))
        allFrames[:,:,0]=im0
        for pic in range(1,nPics):
            imFile=surfDir+'frame'+str(pic)+'.tiff'
            im0=cv2.imread(imFile,-1)
            allFrames[:,:,pic]=im0

        #AVERAGE OVER IMAGES IN FOLDER
        imAvg=np.mean(allFrames,2)

        # #SAVE IMAGE

        outFile=outDir+'frame0.tiff'
        cv2.imwrite(outFile,np.uint16(imAvg))#THIS FILE MUST BE OPENED WITH CV2 MODULE

        outFile=outDir+'16bitSurf.png'
        imAvg=np.true_divide(imAvg,2**12)*(2**16)
        cv2.imwrite(outFile,np.uint16(imAvg))#THIS FILE MUST BE OPENED WITH CV2 MODULE

def get_reference_frame(sourceRoot,animalID,sessID,refRun='run1'):
    #Cesar Echavarria 11/2016
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot is not defined!")
    if sessID is None:
        raise TypeError("sessID is not defined!")
   # print(glob.glob('%s/*'%(sourceRoot)))
    runFolder=glob.glob('%s/%s_%s_%s_*'%(sourceRoot,animalID,sessID,refRun))
    frameFolder=runFolder[0]+"/frames/"
        
    imFile=frameFolder+'frame0.tiff'
    imRef=misc.imread(imFile)
    return imRef

def get_condition_list(sourceRoot,animalID,sessID,runList):
      #MAKE SURE YOU GET SOME ARGUMENTS
      if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
      if sessID is None:
        raise TypeError("sessID not specified!")
      if runList is None:
        raise TypeError("runList not specified!")

      condList=np.zeros(len(runList))
      for idx,run in enumerate(runList):

        #DEFINE DIRECTORIES
        runFolder=glob.glob('%s/%s_%s_%s_*'%(sourceRoot,animalID,sessID,run))

        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"
        frameTimes,frameCond,frameCount = get_frame_times(planFolder)
        condList[idx]=frameCond[0]
      return condList

 # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# QUALITY CONTROL FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_quality_control_figure_path(qualityControlRoot,\
    motionCorrection=False, smoothing_fwhm=False):
    #Cesar Echavarria 11/2016

    imgOperationDir=''
    #DEFINE DIRECTORIES
    if motionCorrection:
        imgOperationDir=imgOperationDir+'motionCorrection_'

    if smoothing_fwhm is not False:
        imgOperationDir=imgOperationDir+'smoothing_fwhm'+str(smoothing_fwhm)
    else:
        imgOperationDir=imgOperationDir+'noSmoothing'

    QCtargetFolder=qualityControlRoot+'/'+imgOperationDir+'/Figures/'

    return QCtargetFolder


def get_first_frame_correlation(sourceRoot,animalID,sessID,runList):
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if runList is None:
        raise TypeError("runList not specified!")

    for (runCount,run) in enumerate(runList):

        runFolder=glob.glob(sourceRoot+'/'+animalID+'_'+sessID+'_'+str(run)+'_*')
        frameFolder=runFolder[0]+"/frames/"

        # READ IN FRAMES
        imFile=frameFolder+'frame0.tiff'
        im0=misc.imread(imFile)
        sz=im0.shape

        #STORE PIXEL VALUES OF FIRST FRAME
        if runCount==0:
            frame1PixMat=np.zeros((sz[0]*sz[1],len(runList)))
        frame1PixMat[:,runCount]=np.reshape(im0,sz[0]*sz[1])
    R=np.corrcoef(np.transpose(frame1PixMat))   
    
    return R

def quick_quality_control(sourceRoot, targetRoot, animalID, sessID, runList):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if runList is None:
        raise TypeError("runList not specified!")

        
    # DEFINE DIRECTORIES

    qualityControlRoot=targetRoot+'/QualControl'
    QCtargetFolder=qualityControlRoot+'/quick/'

    if not os.path.exists(QCtargetFolder):

        #GO THROUGH RUNS AND GET SOME FRAME VALUES
        for (runCount,run) in enumerate(runList):
            print(sourceRoot)
            print(glob.glob(sourceRoot+'/'))

            print(run)

            runFolder=glob.glob(sourceRoot+'/'+animalID+'_'+sessID+'_'+str(run)+'_*')
            print(runFolder)

            frameFolder=runFolder[0]+"/frames/"
            planFolder=runFolder[0]+"/plan/"

            # READ IN FRAME TIMES FILE
            frameTimes,frameCond,frameCount=get_frame_times(planFolder)
            frameTimes=frameTimes[0::100]


            
            print('Loading frames...')

            #GET REFFERNCE FRAME
            imRef=get_reference_frame(sourceRoot,animalID,sessID,runList[0])
            szY,szX=imRef.shape

            # READ IN FRAMES
            frameArray=np.zeros((szY,szX,frameCount))
            for f in range (0,frameCount,100):
                imFile=frameFolder+'frame'+str(f)+'.tiff'
                im0=misc.imread(imFile)
                frameArray[:,:,f]=np.copy(im0)
            frameArray=frameArray[:,:,0::100]


            frameArray=np.reshape(frameArray,(szY*szX,np.shape(frameArray)[2]))

            if not os.path.exists(QCtargetFolder):
                os.makedirs(QCtargetFolder)

            meanF=np.squeeze(np.mean(frameArray,0))#average over pixels

            fig=plt.figure()
            plt.plot(frameTimes,meanF)

            fig.suptitle('Mean Pixel Value Over Time', fontsize=20)
            plt.xlabel('Time (secs)',fontsize=16)
            plt.ylabel('Mean Pixel Value',fontsize=16)
            plt.savefig(QCtargetFolder+sessID+'_'+run+'_meanPixelValue.png')
            plt.close()

            randPix=np.random.randint(0,szY*szX)

            fig=plt.figure()
            plt.plot(frameTimes,frameArray[randPix,:])

            fig.suptitle('Pixel '+str(randPix)+' Value Over Time', fontsize=20)
            plt.xlabel('Time (secs)',fontsize=16)
            plt.ylabel('Mean Pixel Value',fontsize=16)
            plt.savefig(QCtargetFolder+sessID+'_'+run+'_randomPixelValue.png')
            plt.close()

        R=get_first_frame_correlation(sourceRoot,animalID, sessID,runList)
        fig=plt.figure()
        plt.imshow(R,interpolation='none')
        plt.colorbar()
        plt.savefig(QCtargetFolder+sessID+'_firstFrame_CorrelationMatrix.png')
        plt.close()
            


# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# MOTION CORRECTION FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

def motion_registration(imRef,frameStack):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if imRef is None:
        raise TypeError("imRef (reference image) is not defined!")
    if frameStack is None:
        raise TypeError("no frame stack defined!")


    frameCount=np.shape(frameStack)[2]

    imRef_gray=np.uint8(np.true_divide(imRef,np.max(imRef))*255)
    imRef_smooth=cv2.GaussianBlur(imRef_gray, (11,11), .9, .9)
    imRef_forReg = get_gradient(imRef_smooth)
    
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Specify the number of iterations.
    number_of_iterations = 10;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    warpMatrices=np.zeros((2,3,frameCount))
    motionMag=np.zeros((frameCount))
    
    for f in range (0,frameCount):
        if f%1000==0:
            print('Motion Registration at frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        im0_gray=np.uint8(np.true_divide(im0,np.max(im0))*255)
        im0_smooth=cv2.GaussianBlur(im0_gray, (11,11), .9, .9)
        im0_forReg = get_gradient(im0_smooth)
        

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)


        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (imRef_forReg,im0_forReg,warp_matrix, warp_mode, criteria)
        warpMatrices[:,:,f]=warp_matrix
        motionMag[f]=np.sum(np.square(np.eye(2, 3, dtype=np.float32)-warp_matrix))
    print(np.argmax(motionMag)) 
    
    return warpMatrices,motionMag

def apply_motion_correction(frameStack=None,warpMatrices=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack defined!")
    if frameStack is None:
        raise TypeError("no warp matrices provided!")
        
  
        
    #APPLY TRANSFORMATION AND TRIM
    szY,szX,frameCount=np.shape(frameStack)
    newFrameStack=np.zeros((szY,szX,frameCount))
    
    for f in range (0,frameCount):
        if f%1000==0:
            print('Motion Correction at frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        warpMatrix=warpMatrices[:,:,f]
        im1 = cv2.warpAffine(im0, warpMatrix, (szX,szY), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        newFrameStack[:,:,f]=im1
    return newFrameStack



def get_boundaries(img=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if img is None:
        raise TypeError("no image array given!")
    
    szY,szX = np.shape(img)
    theta = get_theta_array(szY,szX)
    
    
    halfAngle=42.5
    physicalUp=np.logical_and(theta>-90-halfAngle, theta <-90+halfAngle)
    physicalRight=np.logical_and(theta>-halfAngle, theta <halfAngle)
    physicalDown=np.logical_and(theta>90-halfAngle, theta <90+halfAngle)
    physicalLeft=np.abs(theta)>180-halfAngle
    

    zeroUp=np.where(np.logical_and(img<10,physicalUp))[0]
    zeroDown=np.where(np.logical_and(img<10,physicalDown))[0]
    zeroLeft=np.where(np.logical_and(img<10,physicalLeft))[1]
    zeroRight=np.where(np.logical_and(img<10,physicalRight))[1]

    if np.size(zeroUp)==0:
        edgeUp=2
    else:
        edgeUp=np.max(zeroUp)+3

    if np.size(zeroDown)==0:
        edgeDown=szY-2
    else:
        edgeDown=np.min(zeroDown)-3

    if np.size(zeroLeft)==0:
        edgeLeft=2
    else:
        edgeLeft=np.max(zeroLeft)+3

    if np.size(zeroRight)==0:
        edgeRight=szX-2
    else:
        edgeRight=np.min(zeroRight)-3
        
    return edgeUp,edgeDown,edgeLeft,edgeRight

def get_motion_corrected_boundaries(frameStack=None,warpMatrices=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack given!")
    if warpMatrices is None:
        raise TypeError("no warp matrix stack given!")

    print('Getting motion corrected image boundaries...')
    szY,szX,frameCount=np.shape(frameStack) 
    
    edgeUpList=np.zeros((frameCount))
    edgeDownList=np.zeros((frameCount))
    edgeLeftList=np.zeros((frameCount))
    edgeRightList=np.zeros((frameCount))
    for f in range (0,frameCount):
        im0=np.copy(frameStack[:,:,f])
        
        edgeUp,edgeDown,edgeLeft,edgeRight = get_boundaries(im0)

        edgeUpList[f]=edgeUp
        edgeDownList[f]=edgeDown
        edgeLeftList[f]=edgeLeft
        edgeRightList[f]=edgeRight
    
    return edgeUpList,edgeDownList,edgeLeftList,edgeRightList
    

def apply_motion_correction_boundaries(frameStack=None,boundaries=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack given!")
    if boundaries is None:
        raise TypeError("no boundaries given!")
    
    print('Applying motion corrected image boundaries...')
    frameCount=np.shape(frameStack)[2]
    
    edgeUp=boundaries[0].astype('int64')
    edgeDown=boundaries[1].astype('int64')
    edgeLeft=boundaries[2].astype('int64')
    edgeRight=boundaries[3].astype('int64')
    
    newSzY=edgeDown-edgeUp
    newSzX=edgeRight-edgeLeft
    
    newFrameStack=np.zeros((newSzY,newSzX,frameCount))
    for f in range (0,frameCount):
        if f%1000==0:
            print('frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        newFrameStack[:,:,f]=im0[edgeUp:edgeDown,edgeLeft:edgeRight]
        
    return newFrameStack

def perform_motion_registration(sourceRoot,targetRoot,animalID, sessID,runList,refRun='run1',saveFrames=True,makeMovies=False,frameRate=None):
    #Cesar Echavarria 11/2016
    
     #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if makeMovies and frameRate is None:
        raise TypeError("user input to make movies but frame rate not specified!")
        
    #DEFINE AND MAKE DIRECTORIES
    motionDir=targetRoot+'/Motion/';
    if not os.path.exists(motionDir):
                os.makedirs(motionDir)

    motionFileDir=motionDir+'Registration/'
    if not os.path.exists(motionFileDir):
                os.makedirs(motionFileDir)

    motionFigDir=motionDir+'Figures/'
    if not os.path.exists(motionFigDir):
                os.makedirs(motionFigDir)
            
    motionMovieDir=motionDir+'Movies/'
    if makeMovies:
        if not os.path.exists(motionMovieDir):
            os.makedirs(motionMovieDir)

    #GET REFFERNCE FRAME
    imRef=get_reference_frame(sourceRoot,animalID,sessID,refRun)
    szY,szX=imRef.shape

    if not os.path.exists(motionFileDir+sessID+'_motionCorrectedBoundaries.npz'):

        #PERFORM REGISTRATION FOR ALL RUNS
        for (runCount,run) in enumerate(runList):
            print('Performing image registration for '+run)

            runFolder=glob.glob(sourceRoot+'/'+animalID+'_'+sessID+'_'+str(run)+'_*')
            frameFolder=runFolder[0]+"/frames/"
            planFolder=runFolder[0]+"/plan/"

            frameTimes,frameCond,frameCount=get_frame_times(planFolder)

            #READ IN FRAMES
            frameArray=np.zeros((szY,szX,frameCount))
            print('Loading frames....')
            for f in range (0,frameCount):
                imFile=frameFolder+'frame'+str(f)+'.tiff'
                im0=misc.imread(imFile)
                frameArray[:,:,f]=im0[:,:]
                
            if makeMovies:
                #GENRATE RAW DATA MOVIE
                frameArrayNorm=normalize_stack(frameArray)
                outFile=motionMovieDir+sessID+'_'+run+'_raw_stack.mp4'
                make_movie_from_stack(motionMovieDir,frameArrayNorm,frameRate,outFile)
        

            #MOTION REGISTRATION
            warpMatrices,motionMag=motion_registration(imRef,frameArray)

            #-> plot motion magnitude (squared error from identity matrix) and save figure
            fig=plt.figure()
            plt.plot(frameTimes,motionMag)
            fig.suptitle('Motion Over Time', fontsize=20)
            plt.xlabel('Time (secs)',fontsize=16)
            plt.ylabel('Motion Magnitude (AU)',fontsize=16)
            plt.savefig(motionFigDir+sessID+'_'+run+'_motionMagnitude.png')
            plt.close()

            #-> save warp matrices
            outFile=motionFileDir+sessID+'_'+run+'_motionRegistration'
            np.savez(outFile,warpMatrices=warpMatrices)

            #APPLY MOTION CORRECTION AND SAVE
            correctedFrameArray=apply_motion_correction(frameArray,warpMatrices)
            if saveFrames:
                outFile=motionFileDir+sessID+'_run'+str(run)+'_correctedFrames'
                np.savez(outFile,correctedFrameArray=correctedFrameArray)

            dummyArray=np.ones(frameArray.shape)*np.mean(frameArray[:])
            correctedDummyArray=apply_motion_correction(dummyArray,warpMatrices)

            #GET MOTION CORRECTED BOUNDARIES
            edgeUpList,edgeDownList,edgeLeftList,edgeRightList = get_motion_corrected_boundaries(correctedDummyArray,warpMatrices)

            if runCount==0:
                if runList[0] == refRun:
                    edgeUp=np.max(edgeUpList)
                    edgeDown=np.min(edgeDownList)
                    edgeLeft=np.max(edgeLeftList)
                    edgeRight=np.min(edgeRightList)
                else:
                    #LOAD BOUNDARIES
                    inFile=motionFileDir+sessID+'_motionCorrectedBoundaries_intermediate.npz'
                    f=np.load(inFile)
                    boundaries_tmp=f['boundaries']

                    edgeUp=boundaries_tmp[0]
                    edgeDown=boundaries_tmp[1]
                    edgeLeft=boundaries_tmp[2]
                    edgeRight=boundaries_tmp[3]

                    edgeUp=np.max([edgeUpLast,edpeUpLast])
                    edgeDown=np.min([edgeDownLast,np.min(edgeDownList)])
                    edgeLeft=np.max([edgeLeftLast,np.max(edgeLeftList)])
                    edgeRight=np.min([edgeRightLast,np.min(edgeRightList)])
            else:
                edgeUp=np.max([edgeUp,np.max(edgeUpList)])
                edgeDown=np.min([edgeDown,np.min(edgeDownList)])
                edgeLeft=np.max([edgeLeft,np.max(edgeLeftList)])
                edgeRight=np.min([edgeRight,np.min(edgeRightList)])

            boundaries=(edgeUp,edgeDown,edgeLeft,edgeRight)  
            #->save boundaries
            outFile=motionFileDir+sessID+'_motionCorrectedBoundaries_intermediate'
            np.savez(outFile,boundaries=boundaries)

            if makeMovies:
                #APPLY BOUNDARIES
                tmp_boundaries=(edgeUp,edgeDown,edgeLeft,edgeRight)  
                trimCorrectedFrameArray = apply_motion_correction_boundaries(correctedFrameArray,tmp_boundaries)

                #GENRATE MOTION CORRECTED MOVIE
                frameArrayNorm=normalize_stack(trimCorrectedFrameArray)
                outFile=motionMovieDir+sessID+'_'+run+'_MC_trimmed_stack.mp4'
                make_movie_from_stack(motionMovieDir,frameArrayNorm,frameRate,outFile)

        boundaries=(edgeUp,edgeDown,edgeLeft,edgeRight)  
        #->save boundaries
        outFile=motionFileDir+sessID+'_motionCorrectedBoundaries'
        np.savez(outFile,boundaries=boundaries)
        


# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# VISUALIZATION

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

        
def make_movie_from_stack(rootDir,frameStack,frameRate=24,movFile='test.mp4'):
    #Cesar Echavarria 10/2016
    
    #CHECK INPUTS
    if frameStack is None:
        raise TypeError("no frame stack provided!")
    if np.amax(frameStack) > 1:
        raise TypeError("frame stack values must be in the range 0-1")
    
    
    #GET STACK INFO
    szY,szX,nFrames=np.shape(frameStack)



    #MAKE TEMP FOLDER
    tmpDir=rootDir+'/tmp/'
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)


    #WRITE TO TEMP FOLDER
    for i in range(0,nFrames):
        outFile=tmpDir+str(i)+'.png'
        frame0=frameStack[:,:,i]
        if szY%2 == 1:
            frame0=np.delete(frame0,0,0)
        if szX%2 == 1:
            frame0=np.delete(frame0,0,1)

        frame=np.uint8(frame0*255)
        misc.imsave(outFile,frame)

    #WRITE VIDEO
    cmd='ffmpeg -y -r '+'%.3f'%frameRate+' -i '+tmpDir+'%d.png -vcodec libx264 -f mp4 -pix_fmt yuv420p '+movFile
    os.system(cmd)


    #GET RID OF TEMP FOLDER
    shutil.rmtree(tmpDir)
 




def visualize_single_run(sourceRoot, targetRoot, animalID, sessID, runList, smooth_fwhm=None, magRatio_thresh=None,\
    analysisDir=None,motionCorrection=False, flip = False, modify_range=True, mask =None):

    anatSource=os.path.join(targetRoot,'Surface')
    motionDir=os.path.join(targetRoot,'Motion')
    motionFileDir=os.path.join(motionDir, 'Registration')


    fileInDir=os.path.join(analysisDir,'SingleRunData','Files')
    figOutDirRoot=os.path.join(analysisDir,'SingleRunData','Figures')
    fileOutDirRoot=os.path.join(analysisDir,'SingleRunData','Files')
    #for file name
    smoothString=''
    threshString=''

    condList = get_condition_list(sourceRoot,animalID,sessID,runList)

    # runCount = 0
    # run = runList[runCount]

    for runCount,run in enumerate(runList):
        print('Current Run: %s'%(run))
        cond = condList[runCount]
        figOutDir=os.path.join(figOutDirRoot,'cond%s'%(str(int(cond))))
        if not os.path.exists(figOutDir):
            os.makedirs(figOutDir)

        #LOAD MAPS
        fileName = '%s_%s_map.npz'%(sessID, run)
        f=np.load(os.path.join(fileInDir,fileName))
        phaseMap=f['phaseMap']
        magRatioMap=f['magRatioMap']

        if smooth_fwhm is not None:
            phaseMap=smooth_array(phaseMap,smooth_fwhm,phaseArray=True)
            magRatioMap=smooth_array(magRatioMap,smooth_fwhm)
            smoothString='_fwhm_'+str(smooth_fwhm)


        #set phase map range for visualization
        if modify_range:
            phaseMapDisplay=np.copy(phaseMap)
            phaseMapDisplay[phaseMap<0]=-phaseMap[phaseMap<0]
            phaseMapDisplay[phaseMap>0]=(2*np.pi)-phaseMap[phaseMap>0]

            rangeMin=0
            rangeMax=2*np.pi
        else:
            phaseMapDisplay=np.copy(phaseMap)
            rangeMin=-np.pi
            rangeMax=np.pi


        #apply threshhold
        if magRatio_thresh is not None:
            phaseMapDisplay[magRatioMap<magRatio_thresh]=np.nan
            threshString='_thresh_'+str(magRatio_thresh)
        else:
            magRatiothresh = np.max(magRatioMap)
            phaseMapDisplay[magRatioMap<magRatio_thresh]=np.nan
            threshString='_thresh_'+str(magRatio_thresh)

        #load surface for overlay
        #READ IN SURFACE

        
        imFile=anatSource+'/frame0_registered.tiff'
        if not os.path.isfile(imFile):
            imFile=anatSource+'/frame0.tiff'

        imSurf=cv2.imread(imFile,-1)
        szY,szX=imSurf.shape
        imSurf=np.true_divide(imSurf,2**12)*2**8

        if flip:
            print('Flipping Images')
            imSurf = np.fliplr(imSurf)
            phaseMapDisplay = np.fliplr(phaseMapDisplay)

        if motionCorrection:
            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+'/'+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']
            padDown=int(boundaries[0])
            padUp=int(szY-boundaries[1])
            padLeft=int(boundaries[2])
            padRight=int(szX-boundaries[3])

            phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((np.nan, np.nan),(np.nan,np.nan)))
        #plot
        fileName = 'overlay_images_%s_%s%s%s.png'%(sessID,run,smoothString,threshString)


        dpi = 80
        szY,szX = imSurf.shape
        # What size does the figure need to be in inches to fit the image?
        figsize = szX / float(dpi), szY / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis('off')

        ax.imshow(imSurf, 'gray')
        ax.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=rangeMin,vmax=rangeMax)

        fig.savefig(os.path.join(figOutDir,fileName), dpi=dpi, transparent=True)
        plt.close()

        #output masked image as well, if indicated
        if mask is not None:
            #load mask
            maskFile=targetRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
            f=np.load(maskFile)
            maskM=f['maskM']

            #apply mask
            phaseMapDisplay[maskM==0]=np.nan

            #plot
            outFile=outFile = '%s_%s%s%s_phaseMap_mask_%s_image.png'%\
            (figOutDir+sessID,run,smoothString,threshString,mask)

            #Create a figure of the right size with one axes that takes up the full figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])

            # Hide spines, ticks, etc.
            ax.axis('off')
            ax.imshow(imSurf, 'gray')
            ax.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=rangeMin,vmax=rangeMax)

            fig.savefig(outFile, dpi=dpi, transparent=True)
            plt.close()

def visualize_average_run(sourceRoot, targetRoot, animalID, sessID, runList, smooth_fwhm=None, magRatio_thresh=None,\
    analysisDir=None,motionCorrection=False, flip = False, modify_range=True, mask =None):
        anatSource=os.path.join(targetRoot,'Surface')
        motionDir=os.path.join(targetRoot,'Motion')
        motionFileDir=os.path.join(motionDir, 'Registration')


        fileInDir=os.path.join(analysisDir,'phase_decoding','Files')
        figOutDirRoot=os.path.join(analysisDir,'phase_decoding','Figures')
        fileOutDirRoot=os.path.join(analysisDir,'phase_decoding','Files')


        smoothString=''
        threshString=''

        condList = get_condition_list(sourceRoot,animalID, sessID,runList)


        for condCount,cond in enumerate(np.unique(condList)):
            print('Current Condition: %s'%(cond))
            cond = condList[condCount]
            figOutDir=os.path.join(figOutDirRoot,'cond%s'%(str(int(cond))))
            if not os.path.exists(figOutDir):
                os.makedirs(figOutDir)

            #LOAD MAPS
            fileName = '%s_cond%i_maps.npz'%(sessID, cond)
            f=np.load(os.path.join(fileInDir,fileName))
            phaseMap=f['phaseMap']
            magRatioMap=f['magRatioMap']

            if smooth_fwhm is not None:
                phaseMap=smooth_array(phaseMap,smooth_fwhm,phaseArray=True)
                magRatioMap=smooth_array(magRatioMap,smooth_fwhm)
                smoothString='_fwhm_'+str(smooth_fwhm)


            #set phase map range for visualization
            if modify_range:
                phaseMapDisplay=np.copy(phaseMap)
                phaseMapDisplay[phaseMap<0]=-phaseMap[phaseMap<0]
                phaseMapDisplay[phaseMap>0]=(2*np.pi)-phaseMap[phaseMap>0]

                rangeMin=0
                rangeMax=2*np.pi
            else:
                phaseMapDisplay=np.copy(phaseMap)
                rangeMin=-np.pi
                rangeMax=np.pi


            #apply threshhold
            if magRatio_thresh is not None:
                phaseMapDisplay[magRatioMap<magRatio_thresh]=np.nan
                threshString='_thresh_'+str(magRatio_thresh)
            else:
                magRatiothresh = np.max(magRatioMap)
                phaseMapDisplay[magRatioMap<magRatio_thresh]=np.nan
                threshString='_thresh_'+str(magRatio_thresh)


            #load surface for overlay
            #READ IN SURFACE


            imFile=anatSource+'/frame0_registered.tiff'
            if not os.path.isfile(imFile):
                imFile=anatSource+'/frame0.tiff'

            imSurf=cv2.imread(imFile,-1)
            szY,szX=imSurf.shape
            imSurf=np.true_divide(imSurf,2**12)*2**8

            if flip:
                print('Flipping Images')
                imSurf = np.fliplr(imSurf)
                phaseMapDisplay = np.fliplr(phaseMapDisplay)

            if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+'/'+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])

                phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((np.nan, np.nan),(np.nan,np.nan)))
            #plot
            fileName = 'overlay_images_%s_cond%s%s%s.png'%(sessID,str(int(cond)),smoothString,threshString)


            dpi = 80
            szY,szX = imSurf.shape
            # What size does the figure need to be in inches to fit the image?
            figsize = szX / float(dpi), szY / float(dpi)

            # Create a figure of the right size with one axes that takes up the full figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])

            # Hide spines, ticks, etc.
            ax.axis('off')

            ax.imshow(imSurf, 'gray')
            ax.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=rangeMin,vmax=rangeMax)

            fig.savefig(os.path.join(figOutDir,fileName), dpi=dpi, transparent=True)
            plt.close()

            #output masked image as well, if indicated
            if mask is not None:
                #load mask
                maskFile=targetRoot+'/masks/Files/'+mask+'.npz'
                f=np.load(maskFile)
                maskM=f['maskM']

                #apply mask
                phaseMapDisplay[maskM==0]=np.nan

                #plot
                outFile=outFile = '%s_cond%s%s%s_phaseMap_mask_%s_image.png'%\
                (figOutDir+sessID,str(int(cond)),smoothString,threshString,mask)

                #Create a figure of the right size with one axes that takes up the full figure
                fig = plt.figure(figsize=figsize)
                ax = fig.add_axes([0, 0, 1, 1])

                # Hide spines, ticks, etc.
                ax.axis('off')
                ax.imshow(imSurf, 'gray')
                ax.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=rangeMin,vmax=rangeMax)

                fig.savefig(outFile, dpi=dpi, transparent=True)
                plt.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# PERIODIC STIM CODE

# # # # # # # # # # # # # # # # # # # # # # # # # # # #



def exclude_edge_periods(frameArray,frameTimes,stimFreq):
    tStart=np.true_divide(1,stimFreq)
    nCycles=np.round(frameTimes[-1]/np.true_divide(1,stimFreq))
    tEnd=np.true_divide(1,stimFreq)*(nCycles-1)
    
    indStart=np.where(frameTimes>tStart)[0][0]
    indEnd=np.where(frameTimes>tEnd)[0][0]-1
    
    arrayNew=frameArray[:,indStart:indEnd]
    frameTimesNew=frameTimes[indStart:indEnd]
    
    return arrayNew,frameTimesNew

def get_analysis_path_phase(analysisRoot, targetFreq, interp=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False,averageFrames=None,smooth_fwhm = None):
    #Cesar Echavarria 2/2017

    interpString=''
    excludeString=''
    averageString=''
    removeRollingString=''
    fwhmString = ''
    #DEFINE DIRECTORIES
    if motionCorrection:
        imgOperationDir='motion_corrected'
    else:
        imgOperationDir='not_motion_corrected'


    procedureDir=''
    if interp:
        interpString='interpolate_'
          
    if excludeEdges:
        excludeString='excludeEdges_'

    if averageFrames is not None:
        averageString='averageFrames_'+str(averageFrames)+'_'

    if removeRollingMean:
        removeRollingString='minusRollingMean'
    if smooth_fwhm:
    	fwhmString = 'fwhm_%i'%(int(smooth_fwhm))

    procedureDir=interpString+excludeString+averageString+fwhmString+removeRollingString

    phaseDir='%s/phase/%s/%s/targetFreq_%sHz/'%(analysisRoot,imgOperationDir,procedureDir,targetFreq)


    return phaseDir


def analyze_periodic_data_per_run(sourceRoot, targetRoot, animalID,sessID, runList, stimFreq, frameRate, \
    interp=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False,averageFrames=None,loadCorrectedFrames=True,saveImages=True,makeMovies=True,\
    stimType=None,mask=None):
     # DEFINE DIRECTORIES
    anatSource=os.path.join(targetRoot,'Surface')
    motionDir=os.path.join(targetRoot,'Motion')
    motionFileDir=os.path.join(motionDir, 'Registration')


    analysisRoot = os.path.join(targetRoot,'Analyses')
    analysisDir=get_analysis_path_phase(analysisRoot, stimFreq, interp, excludeEdges,removeRollingMean, \
        motionCorrection,averageFrames)
    fileOutDir=os.path.join(analysisDir,'SingleRunData','Files')
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    if saveImages:
        figOutDirRoot=os.path.join(analysisDir,'SingleRunData','Figures')



    condList = get_condition_list(sourceRoot,animalID,sessID,runList)
    if interp:
        #GET INTERPOLATION TIME
        newStartT,newEndT=get_interp_extremes(sourceRoot,animalID,sessID,runList,stimFreq)
        newTimes=np.arange(newStartT+(1.0/frameRate),newEndT,1.0/frameRate)#always use the same time points


    for runCount,run in enumerate(runList):
        print('Current Run: %s'%(run))
        #DEFINE DIRECTORIES

        runFolder=glob.glob(sourceRoot+'/'+animalID+'_'+sessID+'_'+str(run)+'_*')
        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"


        # READ IN FRAME TIMES FILE
        frameTimes,frameCond,frameCount=get_frame_times(planFolder)
        cond=frameCond[0]

        if saveImages:
            figOutDir=os.path.join(figOutDirRoot,'cond%s'%(cond))
            if not os.path.exists(figOutDir):
                os.makedirs(figOutDir)

        #READ IN FRAMES
        print('Loading frames...')
        if motionCorrection:

            if loadCorrectedFrames:
                #LOAD MOTION CORRECTED FRAMES
                inFile=motionFileDir+'/'+sessID+'_'+run+'_correctedFrames.npz'
                f=np.load(inFile)
                frameArray=f['correctedFrameArray']
            else:
                #GET REFFERNCE FRAME
                imRef=get_reference_frame(sourceRoot,animalID,sessID,runList[0])
                szY,szX=imRef.shape

                # READ IN FRAMES
                frameArray=np.zeros((szY,szX,frameCount))
                for f in range (0,frameCount):
                    imFile=frameFolder+'frame'+str(f)+'.tiff'
                    im0=misc.imread(imFile)
                    frameArray[:,:,f]=np.copy(im0)

                #-> load warp matrices
                inFile=motionFileDir+'/'+sessID+'_'+run+'_motionRegistration.npz'
                f=np.load(inFile)
                warpMatrices=f['warpMatrices']

                #APPLY MOTION CORRECTION
                frameArray=apply_motion_correction(frameArray,warpMatrices)

            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+'/'+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']

            #APPLY BOUNDARIES
            frameArray = apply_motion_correction_boundaries(frameArray,boundaries)

            szY,szX = np.shape(frameArray[:,:,0])


        else:
            #GET REFFERNCE FRAME
            imRef=get_reference_frame(sourceRoot,animalID,sessID,runList[0])
            szY,szX=imRef.shape

            # READ IN FRAMES
            frameArray=np.zeros((szY,szX,frameCount))
            for f in range (0,frameCount):
                imFile=frameFolder+'frame'+str(f)+'.tiff'
                im0=misc.imread(imFile)
                frameArray[:,:,f]=np.copy(im0)

        frameArray=np.reshape(frameArray,(szY*szX,frameCount))

        #INTERPOLATE FOR CONSTANT FRAME RATE
        if interp:
            print('Interpolating....')
            frameArray=interpolate_array(frameTimes,frameArray,newTimes)
            frameTimes=newTimes

        #EXCLUDE FIRST AND LAST PERIOD
        if excludeEdges:
                print('Excluding first and last periods...')
                frameArray,frameTimes=exclude_edge_periods(frameArray,frameTimes,stimFreq)

        meanPixelValue=np.mean(frameArray,1)

        # REMOVE ROLLING AVERAGE
        if removeRollingMean:

            print('Removing rolling mean....')
            detrendedFrameArray=np.zeros(np.shape(frameArray))
            rollingWindowSz=int(np.ceil((np.true_divide(1,stimFreq)*2)*frameRate))

            for pix in range(0,np.shape(frameArray)[0]):

                tmp0=frameArray[pix,:];
                tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                rollingAvg=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                rollingAvg=rollingAvg[rollingWindowSz:-rollingWindowSz]


                detrendedFrameArray[pix,:]=np.subtract(tmp0,rollingAvg)
            frameArray=detrendedFrameArray
            del detrendedFrameArray

        #AVERAGE FRAMES
        if averageFrames is not None:
            print('Pooling frames...')
            smoothFrameArray=np.zeros(np.shape(frameArray))
            rollingWindowSz=averageFrames

            for pix in range(0,np.shape(frameArray)[0]):

                tmp0=frameArray[pix,:];
                tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                tmp2=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                tmp2=tmp2[rollingWindowSz:-rollingWindowSz]

                smoothFrameArray[pix,:]=tmp2
            frameArray=smoothFrameArray
            del smoothFrameArray

        #Get FFT
        print('Analyzing phase and magnitude....')
        fourierData = np.fft.fft(frameArray)
        #Get magnitude and phase data
        magData=abs(fourierData)
        phaseData=np.angle(fourierData)

        signalLength=np.shape(frameArray)[1]
        freqs = np.fft.fftfreq(signalLength, float(1/frameRate))
        idx = np.argsort(freqs)

        freqs=freqs[idx]
        magData=magData[:,idx]
        phaseData=phaseData[:,idx]

        freqs=freqs[np.round(signalLength/2)+1:]#excluding DC offset
        magData=magData[:,np.round(signalLength/2)+1:]#excluding DC offset
        phaseData=phaseData[:,np.round(signalLength/2)+1:]#excluding DC offset


        freqIdx=np.argmin(np.absolute(freqs-stimFreq))
        topFreqIdx=np.where(freqs>1)[0][0]

        #GET PERCENT SIGNAL MODULATION
        meanPixelValue=np.expand_dims(meanPixelValue,1)
        meanPixelValue=np.tile(meanPixelValue,(1,frameArray.shape[1]))
        frameArrayPSC=np.true_divide(frameArray,meanPixelValue)*100

        #OUTPUT TEXT FILE FREQUENCY CHANNEL ANALYZED
        if runCount == 0:

            outFile=os.path.join(fileOutDir,'frequency_analyzed.txt')
            freqTextFile = open(outFile, 'w+')
            freqTextFile.write(run+' '+str(np.around(freqs[freqIdx],4))+' Hz\n')

        if saveImages:

            maxModIdx=np.argmax(magData[:,freqIdx],0)
            figName = 'magnitude_%s_%s.png'%(sessID,run)
            fig=plt.figure()
            plt.plot(freqs,magData[maxModIdx,:])
            fig.suptitle(sessID+' '+run+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()

            figName = 'magnitude_%s_%s.png'%(sessID,run)
            fig=plt.figure()
            plt.plot(freqs[0:topFreqIdx],magData[maxModIdx,0:topFreqIdx])
            fig.suptitle(sessID+' '+run+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()

            stimPeriod_t=np.true_divide(1,stimFreq)
            stimPeriod_frames=stimPeriod_t*frameRate
            periodStartFrames=np.round(np.arange(0,len(frameTimes),stimPeriod_frames)).astype('int64')

            figName = 'timecourse_%s_%s.png'%(sessID,run)
            fig=plt.figure()

            plt.plot(frameTimes,frameArrayPSC[maxModIdx,:])
            fig.suptitle(sessID+' '+run+' timecourse', fontsize=20)
            plt.xlabel('Time (s)',fontsize=16)
            plt.ylabel('Signal Change (%)',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            if interp:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            else:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            axes.set_xlim([frameTimes[0],frameTimes[-1]])
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()

        #index into target frequecny and reshape arrays to maps
        magArray=magData[:,freqIdx]
        magMap=np.reshape(magArray,(szY,szX))
        phaseArray=phaseData[:,freqIdx]
        phaseMap=np.reshape(phaseArray,(szY,szX))

        #set phase map range for visualization
        phaseMapDisplay=np.copy(phaseMap)
        phaseMapDisplay[phaseMap<0]=-phaseMap[phaseMap<0]
        phaseMapDisplay[phaseMap>0]=(2*np.pi)-phaseMap[phaseMap>0]

        #get various measures of data quality
        #1) magnitude and ratio of magnitude
        tmp=np.copy(magData)
        np.delete(tmp,freqIdx,1)
        nonTargetMagArray=np.sum(tmp,1)
        magRatio=magArray/nonTargetMagArray
        magRatioMap=np.reshape(magRatio,(szY,szX))
        nonTargetMagMap=np.reshape(nonTargetMagArray,(szY,szX))

        #2) amplitude and variance expained

        t=frameTimes*(2*np.pi)*stimFreq
        t=np.transpose(np.expand_dims(t,1))
        tMatrix=np.tile(t,(phaseData.shape[0],1))

        phi=np.expand_dims(phaseArray,1)
        phiMatrix=np.tile(phi,(1,frameArray.shape[1]))
        Xmatrix=np.cos(tMatrix+phiMatrix)

        betaArray=np.zeros((frameArray.shape[0]))
        varExpArray=np.zeros((frameArray.shape[0]))

        for pix in range(frameArray.shape[0]):
            x=np.expand_dims(Xmatrix[pix,:],1)
            y=frameArrayPSC[pix,:]
            beta=np.matmul(np.linalg.pinv(x),y)
            betaArray[pix]=beta
            yHat=x*beta
            SSreg=np.sum((yHat-np.mean(y,0))**2)
            SStotal=np.sum((y-np.mean(y,0))**2)
            varExpArray[pix]=SSreg/SStotal

        betaMap=np.reshape(betaArray,(szY,szX))
        varExpMap=np.reshape(varExpArray,(szY,szX))

        fileName='%s_%s_map'%(sessID,run)
        np.savez(os.path.join(fileOutDir,fileName),phaseMap=phaseMap,magMap=magMap,magRatioMap=magRatioMap,\
                 nonTargetMagMap=nonTargetMagMap,betaMap=betaMap,varExpMap=varExpMap)

        if saveImages:
            figName = 'mag_map_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)

            fig=plt.figure()
            plt.imshow(magMap)
            plt.colorbar()
            fig.suptitle(sessID+' '+run+' magMap', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()

            figName = 'mag_ratio_map_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)
            fig=plt.figure()
            plt.imshow(magRatioMap)
            plt.colorbar()
            fig.suptitle(sessID+' '+run+' magRatioMap', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()


            figName = 'amplitude_map_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)
            fig=plt.figure()
            plt.imshow(betaMap)
            plt.colorbar()
            fig.suptitle(sessID+' '+run+' Amplitude', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()


            figName = 'variance_explained_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)
            fig=plt.figure()
            plt.imshow(varExpMap)
            plt.colorbar()
            fig.suptitle(sessID+' '+run+' Variance Explained', fontsize=14)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()


            figName = 'phase_map_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)
            fig=plt.figure()
            plt.imshow(phaseMapDisplay,'nipy_spectral',vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' '+run+' phaseMap', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()


            #load surface for overlay
            #READ IN SURFACE
            imFile=anatSource+'/frame0_registered.tiff'
            if not os.path.isfile(imFile):
                imFile=anatSource+'/frame0.tiff'

            imSurf=cv2.imread(imFile,-1)
            szY,szX=imSurf.shape
            imSurf=np.true_divide(imSurf,2**12)*2**8

            if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+'/'+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])

                phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

            #plot
            fileName = 'phase_map_overlay_%s_Hz_%s_%s'%(str(np.around(freqs[freqIdx],4)),sessID,run)

            fig=plt.figure()
            plt.imshow(imSurf, 'gray')
            plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()

            #output masked image as well, if indicated
            if mask is not None:
                #load mask
                maskFile=targetRoot+'/'+animalID,'/',+sessID+'/masks/Files/'+mask+'.npz'
                f=np.load(maskFile)
                maskM=f['maskM']

                #apply mask
                phaseMapDisplay=phaseMapDisplay*maskM

                #plot
                outFile = '%s_%s_%sHz_phaseMap_mask_%s.png'%\
                    (figOutDir+sessID,str(run),str(np.around(freqs[freqIdx],4)),mask)
                fig=plt.figure()
                plt.imshow(imSurf, 'gray')
                plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
                plt.colorbar()
                fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
                plt.savefig(outFile)
                plt.close()

            #define legend matrix
            if stimType=='bar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(0, 2*np.pi, szScreenX)
                y = np.linspace(0, 2*np.pi, szScreenX)
                xv, yv = np.meshgrid(x, y)


                if cond==1:
                    legend=xv[296:1064,:]
                elif cond==2:
                    xv=(2*np.pi)-xv
                    legend=xv[296:1064,:]
                elif cond==3:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, legend = np.meshgrid(x, y)

                elif cond==4:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, yv = np.meshgrid(x, y)
                    legend=(2*np.pi)-yv

                figName = '%s_cond%s_legend.png'%(sessID,cond)
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(os.path.join(figOutDir,figName))
                plt.close()
            elif stimType=='polar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(-1, 1, szScreenX)
                y = np.linspace(-1, 1, szScreenX)
                xv, yv = np.meshgrid(x, y)

                rad,theta=cart2pol(xv,yv)

                x = np.linspace(-szScreenX/2, szScreenX/2, szScreenX)
                y = np.linspace(-szScreenY/2, szScreenY/2, szScreenY)
                xv, yv = np.meshgrid(x, y)

                radMask,thetaMask=cart2pol(xv,yv)


                thetaLegend=np.copy(theta)
                thetaLegend[theta<0]=-theta[theta<0]
                thetaLegend[theta>0]=(2*np.pi)-thetaLegend[theta>0]
                if cond == 1:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==2:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    thetaLegend=(2*np.pi)-thetaLegend
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==3:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)


                elif cond ==4:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)
                    legend=(2*np.pi)-legend
                    legend[radMask>szScreenY/2]=0

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()


        freqTextFile.close()

def smooth_phase_array(theta,sigma,sz):
    #build 2D Gaussian Kernel
    kernelX = cv2.getGaussianKernel(sz, sigma); 
    kernelY = cv2.getGaussianKernel(sz, sigma); 
    kernelXY = kernelX * kernelY.transpose(); 
    kernelXY_norm=np.true_divide(kernelXY,np.max(kernelXY.flatten()))
    
    #get x and y components of unit-length vector
    componentX=np.cos(theta)
    componentY=np.sin(theta)
    
    #convolce
    componentX_smooth=signal.convolve2d(componentX,kernelXY_norm,mode='same',boundary='symm')
    componentY_smooth=signal.convolve2d(componentY,kernelXY_norm,mode='same',boundary='symm')

    theta_smooth=np.arctan2(componentY_smooth,componentX_smooth)
    return theta_smooth

def smooth_phase_array(theta,sigma,sz):
    #build 2D Gaussian Kernel
    kernelX = cv2.getGaussianKernel(sz, sigma); 
    kernelY = cv2.getGaussianKernel(sz, sigma); 
    kernelXY = kernelX * kernelY.transpose(); 
    kernelXY_norm=np.true_divide(kernelXY,np.max(kernelXY.flatten()))
    
    #get x and y components of unit-length vector
    componentX=np.cos(theta)
    componentY=np.sin(theta)
    
    #convolce
    componentX_smooth=signal.convolve2d(componentX,kernelXY_norm,mode='same',boundary='symm')
    componentY_smooth=signal.convolve2d(componentY,kernelXY_norm,mode='same',boundary='symm')

    theta_smooth=np.arctan2(componentY_smooth,componentX_smooth)
    return theta_smooth

def smooth_array(inputArray,fwhm,phaseArray=False):
    szList=np.array([None,None,None,11,None,21,None,27,None,31,None,37,None,43,None,49,None,53,None,59,None,55,None,69,None,79,None,89,None,99])
    sigmaList=np.array([None,None,None,.9,None,1.7,None,2.6,None,3.4,None,4.3,None,5.1,None,6.4,None,6.8,None,7.6,None,8.5,None,9.4,None,10.3,None,11.2,None,12])
    sigma=sigmaList[fwhm]
    sz=szList[fwhm]
    if phaseArray:
        outputArray=smooth_phase_array(inputArray,sigma,sz)
    else:
        outputArray=cv2.GaussianBlur(inputArray, (sz,sz), sigma, sigma)
    return outputArray

def smooth_stack(inputStack,fwhm,phaseStack=False):
    outputStack = np.zeros(np.shape(inputStack))
    for f in range(np.shape(inputStack)[2]):
    	outputStack[:,:,f]=smooth_array(np.squeeze(inputStack[:,:,f]),fwhm)
    return outputStack



def get_analysis_path_timecourse(analysisRoot,  interpolate=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False,averageFrames=None):
    #Cesar Echavarria 11/2016

    imgOperationDir=''
    #DEFINE DIRECTORIES
    if motionCorrection:
        imgOperationDir=imgOperationDir+'motion_corrected'
    else:
        imgOperationDir=imgOperationDir+'not_motion_corrected'


    procedureDir=''
    if interpolate:
        procedureDir=procedureDir+'interpolate'
        
    if excludeEdges:
        procedureDir=procedureDir+'excludeEdges'

    if averageFrames is not None:
        procedureDir=procedureDir+'_averageFrames_'+str(averageFrames)


    if removeRollingMean:
        procedureDir=procedureDir+'_minusRollingMean'

    timecourseDir=analysisRoot+'/timecourse/'+imgOperationDir+'/'+procedureDir+'/'

    return timecourseDir

def get_interp_extremes(sourceRoot,animalID,sessID,runList,stimFreq):

    run=runList[0]
    
    runFolder=glob.glob('%s/%s_%s_%s_*'%(sourceRoot,animalID,sessID,run))
    frameFolder=runFolder[0]+"/frames/"
    planFolder=runFolder[0]+"/plan/"
    frameTimes,frameCond,frameCount = get_frame_times(planFolder)

    tMin=np.true_divide(1,stimFreq)
    nCycles=np.round(frameTimes[-1]/np.true_divide(1,stimFreq))
    tMax=np.true_divide(1,stimFreq)*(nCycles-1)
    return tMin, tMax

def interpolate_array(t0,array0,tNew):
    interpF = interpolate.interp1d(t0, array0,1)
    
    arrayNew=np.zeros((array0.shape[0],tNew.size))
    if np.any(tNew<t0[0]):
        arrayNew[:,tNew<t0[0]]=np.expand_dims(array0[:,0],1)
        ind0=np.where(tNew<t0[0])[0][-1]+1
    else:
        ind0=0
    if np.any(tNew>t0[-1]):
        arrayNew[:,tNew>t0[-1]]=np.expand_dims(array0[:,-1],1)
        ind1=np.where(tNew>t0[-1])[0][0]
    else:
        ind1=tNew.size

    arrayNew[:,ind0:ind1]=interpF(tNew[ind0:ind1])
    
    return arrayNew

def analyze_complete_timecourse(sourceRoot, targetRoot, animalID, sessID, runList, stimFreq, frameRate, \
                               interp=False, excludeEdges=False, removeRollingMean=False, \
                               motionCorrection=False,averageFrames=None, groupPeriods = None, loadCorrectedFrames=True,\
                               SDmaps=False,makeSingleRunMovies=False,makeSingleCondMovies=True):


    # DEFINE DIRECTORIES

    motionDir=targetRoot+'/Motion/';
    motionFileDir=motionDir+'Registration/'

    analysisRoot=targetRoot+'/Analyses/'
    analysisDir=get_analysis_path_timecourse(analysisRoot, interp, excludeEdges, removeRollingMean, \
        motionCorrection,averageFrames)

    fileOutDir=analysisDir+'Files/'
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)


    if makeSingleCondMovies:
        avgMovieOutDir=analysisDir+'/Movies/'
        if not os.path.exists(avgMovieOutDir):
            os.makedirs(avgMovieOutDir) 
    if makeSingleRunMovies:
        singleRunMovieOutDir=analysisDir+'/Movies/'
        if not os.path.exists(singleRunMovieOutDir):
            os.makedirs(singleRunMovieOutDir) 
    if SDmaps:
        figOutDir=analysisDir+'/Figures/'
        if not os.path.exists(figOutDir):
            os.makedirs(figOutDir) 

    condList = get_condition_list(sourceRoot,animalID,sessID,runList)
    if interp:
        #GET INTERPOLATION TIME
        newStartT,newEndT=get_interp_extremes(sourceRoot,animalID,sessID,runList,stimFreq)
        newTimes=np.arange(newStartT+(1.0/frameRate),newEndT,1.0/frameRate)#always use the same time points


    for cond in np.unique(condList):
        print('cond='+str(cond))
        condRunList=np.where(condList==cond)[0]
        for counter in range(len(condRunList)):
            print('counter='+str(counter))
            idx=condRunList[counter]
            run = runList[idx]
            print('run='+str(run))
            #DEFINE DIRECTORIES
            runFolder=glob.glob('%s/%s_%s_%s_*'%(sourceRoot,animalID,sessID,run))
            frameFolder=runFolder[0]+"/frames/"
            planFolder=runFolder[0]+"/plan/"
            frameTimes,frameCond,frameCount = get_frame_times(planFolder)

            #READ IN FRAMES
            print('Loading frames...')
            if motionCorrection:

                if loadCorrectedFrames:
                    #LOAD MOTION CORRECTED FRAMES
                    inFile=motionFileDir+sessID+'_'+run+'_correctedFrames.npz'
                    f=np.load(inFile)
                    frameArray=f['correctedFrameArray']
                else:
                    #GET REFFERNCE FRAME
                    imRef=get_reference_frame(sourceRoot,animalID,sessID,runList[0])
                    szY,szX=imRef.shape

                    # READ IN FRAMES
                    frameArray=np.zeros((szY,szX,frameCount))
                    for f in range (0,frameCount):
                        imFile=frameFolder+'frame'+str(f)+'.tiff'
                        im0=misc.imread(imFile)
                        frameArray[:,:,f]=np.copy(im0)

                    #-> load warp matrices
                    inFile=motionFileDir+sessID+'_'+run+'_motionRegistration.npz'
                    f=np.load(inFile)
                    warpMatrices=f['warpMatrices']

                    #APPLY MOTION CORRECTION
                    frameArray=apply_motion_correction(frameArray,warpMatrices)

                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']

                #APPLY BOUNDARIES
                frameArray = apply_motion_correction_boundaries(frameArray,boundaries)

                szY,szX = np.shape(frameArray[:,:,0])


            else:
                #GET REFFERNCE FRAME
                imRef=get_reference_frame(sourceRoot,animalID,sessID,runList[0])
                szY,szX=imRef.shape

                # READ IN FRAMES
                frameArray=np.zeros((szY,szX,frameCount))
                for f in range (0,frameCount):
                    imFile=frameFolder+'frame'+str(f)+'.tiff'
                    im0=misc.imread(imFile)
                    frameArray[:,:,f]=np.copy(im0)

            frameArray=np.reshape(frameArray,(szY*szX,frameCount))
            
            #INTERPOLATE FOR CONSTANT FRAME RATE
            if interp:
                print('Interpolating...')
                frameArray=interpolate_array(frameTimes,frameArray,newTimes)
                frameTimes=newTimes

            if counter == 0:   
                meanPixelValue=np.mean(frameArray,1)
            else:
                meanPixelValue=np.vstack((meanPixelValue,np.mean(frameArray,1)))
            # REMOVE ROLLING AVERAGE
            if removeRollingMean:
                print('Removing Rolling Mean...')
                #INITIALIZE VARIABLES
                detrendedFrameArray=np.zeros(np.shape(frameArray))
                rollingWindowSz=int(np.ceil((np.true_divide(1,stimFreq)*2)*frameRate))

                #REMOVE ROLLING MEAN; KEEP MEAN OFFSET
                for pix in range(0,np.shape(frameArray)[0]):
                    tmp0=frameArray[pix,:]
                    tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                    rollingAvg=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                    rollingAvg=rollingAvg[rollingWindowSz:-rollingWindowSz]


                    detrendedFrameArray[pix,:]=np.subtract(tmp0,rollingAvg)
                frameArray=detrendedFrameArray
                del detrendedFrameArray

            #AVERAGE FRAMES
            if averageFrames is not None:
                print('Pooling frames values....')
                smoothFrameArray=np.zeros(np.shape(frameArray))
                rollingWindowSz=averageFrames

                for pix in range(0,np.shape(frameArray)[0]):

                    tmp0=frameArray[pix,:];
                    tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                    tmp2=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                    tmp2=tmp2[rollingWindowSz:-rollingWindowSz]

                    smoothFrameArray[pix,:]=tmp2
                frameArray=smoothFrameArray
                del smoothFrameArray
            
            #EXCLUDE FIRST AND LAST PERIOD
            if excludeEdges:
                print('Excluding first and last periods...')
                frameArray,frameTimes=exclude_edge_periods(frameArray,frameTimes,stimFreq)

            if makeSingleRunMovies:
                tmp=frameArray
                #RESHAPE
                nPix,nPts=np.shape(tmp)
                tmp=np.reshape(tmp,(szY,szX,nPts))

                #NORMALIZE ARRAY
                tmp=normalize_stack(tmp)

                #MAKE MOVIE
                outFile=singleRunMovieOutDir+'cond'+str(cond)+'_run'+str(counter+1)+'_movie.mp4'
                make_movie_from_stack(singleRunMovieOutDir,tmp,frameRate,outFile)
            

            if counter == 0:
                frameArrayAll=frameArray
            else:
                frameArrayAll=np.dstack((frameArrayAll,frameArray))

            if groupPeriods is not None:
                if counter ==0:
                    stimPeriod=1.0/stimFreq
                    framePeriod=1.0/frameRate
                    nCycles=np.round(frameTimes[-1]/stimPeriod)
                    if excludeEdges:
                        nCycles=nCycles-1
                    nGroups=int(np.floor(nCycles/groupPeriods))
                    print(nCycles,nGroups,groupPeriods)

                for g in range(nGroups):
                    startTime=stimPeriod*((g*groupPeriods)+1)
                    startInd=np.where(frameTimes>startTime)[0][0].astype('int64')
                    endInd=(startInd+np.floor((groupPeriods*stimPeriod)/framePeriod)-1).astype('int64')

                    print(startInd,endInd)
                    frameArrayGroup=frameArray[:,startInd:endInd]
                    if counter == 0 and g == 0:
                        frameArrayGrouped=frameArrayGroup
                    else:
                        frameArrayGrouped=np.dstack((frameArrayGrouped,frameArrayGroup))
                    del frameArrayGroup
                    
            del frameArray


        #AVERAGE PER CONDITION
        meanPixelValue=np.squeeze(np.mean(meanPixelValue,0))
        if len(frameArrayAll.shape)==3:
            frameArrayAvg=np.mean(frameArrayAll,2)
        else:
            frameArrayAvg=frameArrayAll
        del frameArrayAll
        if groupPeriods is None:
            frameArrayGroupedAvg=[]
        else:
            frameArrayGroupedAvg=np.mean(frameArrayGrouped,2)
        
        outFile=fileOutDir+'cond'+str(cond)+'_averageTimeCourse'
        np.savez(outFile,frameArrayAvg=frameArrayAvg,frameTimes=frameTimes,\
                 szY=szY,szX=szX,frameCount=frameCount,\
                 meanPixelValue=meanPixelValue,\
                groupPeriods=groupPeriods,frameArrayGroupedAvg=frameArrayGroupedAvg)


        if SDmaps:
            #GET STANDARD DEV
            pixSD=np.std(frameArrayAvg,1)
            mapSD=np.reshape(pixSD,(szY,szX))
            #MAKE AND SAVE MAP
            fig=plt.figure()
            plt.imshow(mapSD)
            plt.colorbar()
            plt.savefig(figOutDir+'cond'+str(cond)+'_SDmap.png')

        if makeSingleCondMovies:
            if groupPeriods is None:
                #RESHAPE
                nPix,nPts=np.shape(frameArrayAvg)
                frameArrayAvg=np.reshape(frameArrayAvg,(szY,szX,nPts))

                #NORMALIZE ARRAY
                frameArrayAvg=normalize_stack(frameArrayAvg)

                #MAKE MOVIE
                outFile=avgMovieOutDir+'cond'+str(cond)+'_average_movie.mp4'
                make_movie_from_stack_periodic(avgMovieOutDir,frameArrayAvg,stimFreq,frameRate=frameRate,movFile=outFile)
            else:
                 #RESHAPE
                nPix,nPts=np.shape(frameArrayGroupedAvg)
                frameArrayGroupedAvg=np.reshape(frameArrayGroupedAvg,(szY,szX,nPts))

                #NORMALIZE ARRAY
                frameArrayGroupedAvg=normalize_stack(frameArrayGroupedAvg)

                #MAKE MOVIE
                outFile='%scond%s_average_%sPeriods_movie.mp4'%(avgMovieOutDir,str(cond),str(groupPeriods))
                make_movie_from_stack(avgMovieOutDir,frameArrayGroupedAvg,frameRate=frameRate,movFile=outFile)
                
                del frameArrayGroupedAvg
                

        del frameArrayAvg
    
def analyze_periodic_data_from_timecourse(sourceRoot, targetRoot,animalID, sessID, runList, stimFreq, frameRate, \
    interp=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False, averageFrames=None,saveImages=True,mask=None,\
    stimType=None):

    
    analysisRoot=targetRoot+'/Analyses/';
    analysisDir=get_analysis_path_timecourse(analysisRoot, interp, excludeEdges, removeRollingMean, \
    motionCorrection,averageFrames)

    # DEFINE DIRECTORIES
    fileInDir=analysisDir+'Files/'
    anatSource=targetRoot+'/Surface/'

    motionDir=targetRoot+'/Motion/'
    motionFileDir=motionDir+'Registration/'

    analysisDir=analysisDir+'/phase_decoding/'
    fileOutDir=analysisDir+'/Files/'
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    if saveImages:
        figOutDirRoot=analysisDir+'/Figures/'


    #GET CONDITIONS AVAILABLE
    condList = get_condition_list(sourceRoot,animalID,sessID,runList)

    for (condCount,cond) in enumerate(np.unique(condList)):
        print('cond='+str(cond))

        #DEFINE OUTPUT DIRECTORY
        if saveImages:
            figOutDir=figOutDirRoot+'/cond'+str(int(cond))+'/'
            if not os.path.exists(figOutDir):
                os.makedirs(figOutDir)


        #READ IN DATA
        inFile=fileInDir+'cond'+str(cond)+'_averageTimeCourse.npz'
        f=np.load(inFile)
        frameArray=f['frameArrayAvg']
        frameTimes=f['frameTimes']
        szY=f['szY']
        szX=f['szX']
        meanPixelValue=f['meanPixelValue']
        groupPeriods=f['groupPeriods']
        if groupPeriods is not None:
            frameArrayGrouped=f['frameArrayGroupedAvg']


        #Get FFT
        print('Analyzing phase and magnitude....')
        fourierData = np.fft.fft(frameArray)
        #Get magnitude and phase data
        magData=abs(fourierData)
        phaseData=np.angle(fourierData)

        signalLength=np.shape(frameArray)[1]
        freqs = np.fft.fftfreq(signalLength, float(1/frameRate))
        idx = np.argsort(freqs)

        freqs=freqs[idx]
        magData=magData[:,idx]
        phaseData=phaseData[:,idx]

        freqs=freqs[np.round(signalLength/2)+1:]#excluding DC offset
        magData=magData[:,np.round(signalLength/2)+1:]#excluding DC offset
        phaseData=phaseData[:,np.round(signalLength/2)+1:]#excluding DC offset


        freqIdx=np.where(freqs>stimFreq)[0][0]
        topFreqIdx=np.where(freqs>1)[0][0]

        #OUTPUT TEXT FILE FREQUENCY CHANNEL ANALYZED
        if condCount == 0:

            outFile=fileOutDir+'frequency_analyzed.txt'
            freqTextFile = open(outFile, 'w+')
        freqTextFile.write('COND '+str(cond)+' '+str(np.around(freqs[freqIdx],4))+' Hz\n')

        #GET PERCENT SIGNAL MODULATION
        meanPixelValue=np.expand_dims(meanPixelValue,1)
        meanPixelValue=np.tile(meanPixelValue,(1,frameArray.shape[1]))
        frameArrayPSC=np.true_divide(frameArray,meanPixelValue)*100
        if groupPeriods is not None:
            meanPixelValue=meanPixelValue[:,0:frameArrayGrouped.shape[1]]
            frameArrayGroupedPSC=np.true_divide(frameArrayGrouped,meanPixelValue)*100


        if saveImages:
            
            maxModIdx=np.argmax(magData[:,freqIdx],0)
            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_magnitudePlot.png'
            fig=plt.figure()
            plt.plot(freqs,magData[maxModIdx,:])
            fig.suptitle(sessID+' cond '+str(int(cond))+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(outFile)
            plt.close()
            
            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_magnitudePlot_zoom.png'
            fig=plt.figure()
            plt.plot(freqs[0:topFreqIdx],magData[maxModIdx,0:topFreqIdx])
            fig.suptitle(sessID+' cond '+str(int(cond))+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(outFile)
            plt.close()

            stimPeriod_t=np.true_divide(1,stimFreq)
            stimPeriod_frames=stimPeriod_t*frameRate
            periodStartFrames=np.round(np.arange(0,len(frameTimes),stimPeriod_frames)).astype('int64')

            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_timecourse.png'
            fig=plt.figure()
            plt.plot(frameTimes,frameArrayPSC[maxModIdx,])
            fig.suptitle(sessID+' cond '+str(int(cond))+' phase'+str(np.round(np.rad2deg(phaseData[maxModIdx,freqIdx]))), fontsize=10)
            plt.xlabel('Time (s)',fontsize=16)
            plt.ylabel('Signal Change (%)',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            if interp:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            else:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            axes.set_xlim([frameTimes[0],frameTimes[-1]])
            plt.savefig(outFile)
            plt.close()

            if groupPeriods is not None:
                periodStartFramesGrouped=periodStartFrames[0:groupPeriods]
                outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_timecourse_grouped.png'
                fig=plt.figure()
                plt.plot(frameTimes[0:frameArrayGroupedPSC.shape[1]],frameArrayGroupedPSC[maxModIdx,])
                fig.suptitle(sessID+' cond '+str(int(cond))+' phase'+str(np.round(np.rad2deg(phaseData[maxModIdx,freqIdx]))), fontsize=10)
                plt.xlabel('Time (s)',fontsize=16)
                plt.ylabel('Signal Change (%)',fontsize=16)
                axes = plt.gca()
                ymin, ymax = axes.get_ylim()
                if interp:
                    for f in periodStartFramesGrouped:
                        plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
                else:
                    for f in periodStartFramesGrouped:
                        plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
                axes.set_xlim([frameTimes[0],frameTimes[frameArrayGroupedPSC.shape[1]]])
                plt.savefig(outFile)
                plt.close()



        magArray=magData[:,freqIdx]
        magMap=np.reshape(magArray,(szY,szX))
        phaseArray=phaseData[:,freqIdx]
        phaseMap=np.reshape(phaseArray,(szY,szX))

        #set phase map range for visualization
        phaseMapDisplay=np.copy(phaseMap)
        phaseMapDisplay[phaseMap<0]=-phaseMap[phaseMap<0]
        phaseMapDisplay[phaseMap>0]=(2*np.pi)-phaseMap[phaseMap>0]


        #get various measures of data quality
        #1) magnitude and ratio of magnitude
        tmp=np.copy(magData)
        np.delete(tmp,freqIdx,1)
        nonTargetMagArray=np.sum(tmp,1)
        magRatio=magArray/nonTargetMagArray
        magRatioMap=np.reshape(magRatio,(szY,szX))
        nonTargetMagMap=np.reshape(nonTargetMagArray,(szY,szX))

        #2) amplitude and variance expained
        t=frameTimes*(2*np.pi)*stimFreq
        t=np.transpose(np.expand_dims(t,1))
        tMatrix=np.tile(t,(phaseData.shape[0],1))

        phi=np.expand_dims(phaseArray,1)
        phiMatrix=np.tile(phi,(1,frameArray.shape[1]))
        Xmatrix=np.cos(tMatrix+phiMatrix)

        betaArray=np.zeros((frameArray.shape[0]))
        varExpArray=np.zeros((frameArray.shape[0]))

        for pix in range(frameArray.shape[0]):
            x=np.expand_dims(Xmatrix[pix,:],1)
            y=frameArrayPSC[pix,:]
            beta=np.matmul(np.linalg.pinv(x),y)
            betaArray[pix]=beta
            yHat=x*beta
            if pix == maxModIdx:
                signalFit=yHat
            SSreg=np.sum((yHat-np.mean(y,0))**2)
            SStotal=np.sum((y-np.mean(y,0))**2)
            varExpArray[pix]=SSreg/SStotal

        betaMap=np.reshape(betaArray,(szY,szX))
        varExpMap=np.reshape(varExpArray,(szY,szX))

        outFile=fileOutDir+sessID+'_cond'+str(int(cond))+'_maps'
        np.savez(outFile,phaseMap=phaseMap,magMap=magMap,magRatioMap=magRatioMap,\
             nonTargetMagMap=nonTargetMagMap,betaMap=betaMap,varExpMap=varExpMap)



        if saveImages:

            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_timecourse_fit.png'
            fig=plt.figure()
            plt.plot(frameTimes,frameArrayPSC[maxModIdx,],'b')
            plt.plot(frameTimes,signalFit,'r')
            fig.suptitle(sessID+' cond '+str(int(cond))+' phase'+str(np.round(np.rad2deg(phaseData[maxModIdx,freqIdx]))), fontsize=10)
            plt.xlabel('Time (s)',fontsize=16)
            plt.ylabel('Signal Change (%)',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            if interp:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            else:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            axes.set_xlim([frameTimes[0],frameTimes[-1]])
            plt.savefig(outFile)
            plt.close()


            outFile = outFile = '%s_run%s_%s_Hz_magMap.png'%\
            (figOutDir+sessID,str(cond),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(magMap)
            plt.colorbar()#
            fig.suptitle(sessID+' cond'+str(cond)+' magMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = outFile = '%s_run%s_%s_Hz_magRatioMap.png'%\
            (figOutDir+sessID,str(cond),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(magRatioMap)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' magRatioMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = outFile = '%s_run%s_%s_Hz_amplitude.png'%\
            (figOutDir+sessID,str(cond),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(betaMap)
            plt.colorbar()#
            fig.suptitle(sessID+' cond'+str(cond)+' Amiplitude', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = outFile = '%s_run%s_%s_Hz_varianceExplainedMap.png'%\
            (figOutDir+sessID,str(cond),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(varExpMap)
            plt.colorbar()#
            fig.suptitle(sessID+' cond'+str(cond)+' Variance Explained', fontsize=14)
            plt.savefig(outFile)
            plt.close()


            outFile = figOutDir+sessID+'_cond'+str(cond)+'_'+str(np.around(freqs[freqIdx],4))+'Hz_phaseMap.png'
            fig=plt.figure()
            plt.imshow(phaseMapDisplay,'nipy_spectral',vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            #load surface for overlay
            #READ IN SURFACE
            imFile=anatSource+'frame0_registered.tiff'
            if not os.path.isfile(imFile):
                imFile=anatSource+'frame0.tiff'

            imSurf=cv2.imread(imFile,-1)
            szY,szX=imSurf.shape
            imSurf=np.true_divide(imSurf,2**12)*2**8


            if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])

                phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

            #plot
            outFile = '%s_cond%s_%sHz_phaseMap_overlay.png'%\
                (figOutDir+sessID,str(int(cond)),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(imSurf, 'gray')
            plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            #output masked image as well, if indicated
            if mask is not None:
                #load mask
                maskFile=targetRoot+'/masks/Files/'+mask+'.npz'
                f=np.load(maskFile)
                maskM=f['maskM']

                #apply mask
                phaseMapDisplay=phaseMapDisplay*maskM

                #plot
                outFile = '%s_cond%s_%sHz_phaseMap_mask_%s.png'%\
                    (figOutDir+sessID,str(int(cond)),str(np.around(freqs[freqIdx],4)),mask)
                fig=plt.figure()
                plt.imshow(imSurf, 'gray')
                plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
                plt.colorbar()
                fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
                plt.savefig(outFile)
                plt.close()

            #define legend matrix
            if stimType=='bar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(0, 2*np.pi, szScreenX)
                y = np.linspace(0, 2*np.pi, szScreenX)
                xv, yv = np.meshgrid(x, y)


                if cond==1:
                    legend=xv[296:1064,:]
                elif cond==2:
                    xv=(2*np.pi)-xv
                    legend=xv[296:1064,:]
                elif cond==3:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, legend = np.meshgrid(x, y)

                elif cond==4:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, yv = np.meshgrid(x, y)
                    legend=(2*np.pi)-yv

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()
            elif stimType=='polar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(-1, 1, szScreenX)
                y = np.linspace(-1, 1, szScreenX)
                xv, yv = np.meshgrid(x, y)

                rad,theta=cart2pol(xv,yv)

                x = np.linspace(-szScreenX/2, szScreenX/2, szScreenX)
                y = np.linspace(-szScreenY/2, szScreenY/2, szScreenY)
                xv, yv = np.meshgrid(x, y)

                radMask,thetaMask=cart2pol(xv,yv)


                thetaLegend=np.copy(theta)
                thetaLegend[theta<0]=-theta[theta<0]
                thetaLegend[theta>0]=(2*np.pi)-thetaLegend[theta>0]
                if cond == 1:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==2:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    thetaLegend=(2*np.pi)-thetaLegend
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==3:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)
                    
                    
                elif cond ==4:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)
                    legend=(2*np.pi)-legend
                    legend[radMask>szScreenY/2]=0

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()

    freqTextFile.close()

