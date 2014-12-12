#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 





def plotHighestLinkLines(topInds, nodeLocs, channels):
  
    nodes = nodeLocs.shape[0]
    for i in topInds:
        tx, rx, ch = txRxChForLinkNum(i, nodes, channels)
        plt.plot([nodeLocs[tx,0], nodeLocs[rx,0]], 
                 [nodeLocs[tx,1], nodeLocs[rx,1]])
    plt.grid()


# Plot some random link-channels (rows) of the input matrix
def plotLinks(rssDataMat, k):
  
    for i in random.sample(range(rssDataMat.shape[0]), k):
        plt.plot(rssDataMat[i,:])
    plt.grid()
    #plt.xticks(arange(0,690,20))

# Plot the link-channels (rows) of the input matrix with highest "value"
def plotHighestLinks(rssDataMat, value, k, start_ind):
  
    topInds = argsort(value)[-1:-k-1:-1]
    lendata = rssDataMat.shape[1]
    for i in topInds:
        plt.plot(range(start_ind, start_ind+lendata), rssDataMat[i,:])
    plt.grid()
    plt.xlim(start_ind, start_ind+lendata-1)
    plt.xlabel('Sample Time Index', fontsize=18)
    plt.ylabel('Mean-Subtracted RSS (dB)', fontsize=18)
    #plt.legend(('1st Link', '2nd Link', '3rd Link', '4th Link'), loc=0)

    return topInds

# Calculate the t-score, for each link, for the change in mean between 
# the past vs. future window, at each time.
def CalcTScore(rssMat, wl, minStd):

    # Calculate the mean and standard deviation of each wl-length window 
    linkchs, datalen = rssMat.shape
    rssWinAvg   = np.empty((linkchs, datalen-wl))
    rssWinVar   = np.empty((linkchs, datalen-wl))
    for i in range(datalen-wl):
        rssWinAvg[:,i] = np.mean(rssMat[:,i:i+wl], axis=1)
        rssWinVar[:,i] = np.var(rssMat[:,i:i+wl], axis=1)

    # There are two windows, one before the current time, one after.
    # The Welch's t-test: test of a change in mean with unequal variances
    # between the two samples:
    twoGrpT        = np.zeros((linkchs, datalen-wl))
    for i in range(wl, datalen-wl):
        twoGrpStd      = np.sqrt(rssWinVar[:,i-wl] + rssWinVar[:,i])/np.sqrt(wl)
        twoGrpStdwMin  = np.array([max(minStd, v) for v in twoGrpStd])
        twoGrpT[:,i] = (rssWinAvg[:,i-wl] - rssWinAvg[:,i]) / twoGrpStdwMin

    absTSum     = np.mean(np.abs(twoGrpT), axis=0)
    rmsTSum     = np.sqrt(np.mean(np.abs(twoGrpT)**2, axis=0))
    maxT        = np.abs(twoGrpT).max(axis=0)

    return (absTSum, rmsTSum, twoGrpT, maxT)

# Calculate the Mann-Whitney-Wilcoxon U statistic, for each link, for the 
# change in distribution between the past vs. future window, at each time.
def CalcWilcoxonScore(rssMat, wl):

    # Calculate the mean and standard deviation of each wl-length window 
    linkchs, datalen = rssMat.shape

    # There are two windows, one before the current time, one after.
    # The Mann-Whitney-Wilcoxon rank-sum test
    twoGrpZ       = zeros((linkchs, datalen-wl))

    for i in range(wl, datalen-wl):
        for ln in range(linkchs):
            twoGrpZ[ln,i], junk  = stats.ranksums(rssMat[ln, i-wl:i], rssMat[ln, i:i+wl])

    absZSum   = mean(abs(twoGrpZ), axis=0)
    rmsZSum   = sqrt(mean(abs(twoGrpZ)**2, axis=0))

    return (absZSum, rmsZSum)


# Subtract the mean from each row of a 2D array
def removeMeanAndClean(rssMat, noMeastKey):
    linkchs, N = rssMat.shape
    temp       = np.zeros((linkchs, N))
    # Find the mean of each row. Repeat it to make matrix of same size
    for lc in range(linkchs):
        linkr          = rssMat[lc,:]
        
        # RSS values >= noMeastKey mean data was not measured.
        # Do not include noMeastKey values in mean; and set those values to 0.
        missing_data   = np.where(linkr >= noMeastKey)
        linkmean       = np.mean(linkr[linkr < noMeastKey])
        temp[lc, :]    = linkr - linkmean
        temp[lc, missing_data] = 0
    return  temp


# Subtract the mean from each link between start index "si" and end index "ei".
# However, there may be other breakpoints between si and ei.
# Do not calculate the mean across breakpoint boundaries.  
def brokenSubtractMean(rssMat, si, ei, breakpts):
    
    # Create a copy of the array where the mean will be subtracted
    temp     = rssMat[:, si:ei].copy()  
    linkchs  = rssMat.shape[0]
    # Subtract the mean for each link-channel
    for lc in range(linkchs):
        # The complete list of breakpoints includes si, ei, and any 
        # breakpoint between si and ei.  Sort this set to make it a list
        inds = sorted((breakpts[lc] & set(range(si, ei))) | set([si, ei]) )
        # Compute and subtract the mean for each period between breakpoints
        for i in range(len(inds)-1):
            scurr = inds[i] - si
            ecurr = inds[i+1] - si
            temp[lc, scurr:ecurr] -= mean(temp[lc, scurr:ecurr])

    return temp

# Subtract the mean from each link between start index "si" and end index "ei".
# However, there may be other breakpoints between si and ei.
# Do not calculate the mean across breakpoint boundaries.  
# This function uses the same breakpoints across all links.
def brokenSubtractMeanCommon(rssMat, si, ei, breakpts):
    
    # The complete list of breakpoints includes si, ei, and any 
    # breakpoint between si and ei.  Sort this set to make it a list
    inds = sorted((breakpts & set(range(si, ei))) | set([si, ei]) )
    
    # Create a copy of the array where the mean will be subtracted
    temp     = rssMat[:, si:ei].copy()  
    linkchs  = rssMat.shape[0]
    # Subtract the link-channel mean from each link-channel's data
    # Compute and subtract the mean for each period between breakpoints
    for i in range(len(inds)-1):
        scurr = inds[i] - si
        ecurr = inds[i+1] - si
        durationSamps = ecurr-scurr
        mu            = tile(mean(temp[:, scurr:ecurr], axis=1).reshape(linkchs, 
                             1), durationSamps)
        temp[:, scurr:ecurr] -= mu


    return temp


# Return a 1-d array for any size input array
def flatten(inArray):

    outlen = prod(inArray.shape)
    return inArray.reshape(outlen,1)


#####################################################################
#
#  Main
#
#####################################################################


#
# 1. Inputs to the script.
#
dirname      = './'
filename     = 'breathing_2.txt'
startskip    = 460  # Skip the first "startskip" rows
endskip      = 150  # Skip the last "endskip" rows.
plotOption   = True
# if the breathing rate was constant throughout the experiment.
const_breathing_rate = 15.0/60.0  # true breaths per second
noMeastKey   = 127

# Breathing estimator parameters and options
breakptMeanSubtraction = 'false'  # 'common', 'link', or 'false' or 'cusum' or 'wilcoxon'
window_s     = 30       # Duration of time window, seconds
increment_s  = 5        # seconds of window overlap, seconds
Min_RR       = 0.1      # Minimum breathing rate, Hz
Delta_RR     = 0.002    # Granularity of spectrum estimate
Max_RR       = 0.4      # Maximum breathing rate, Hz

plt.ion()


# Parameters you may change:
#   calcSkip:  Rerun the breathing calculation after this many data lines are read
#   buffL:     buffer length, ie, how much data to plot.
#   startSkip: A serial port seems to have a "memory" of several lines,
#              which were saved from the previous experiment run.
#              ** Must be greater than 0.
plotSkip    = 30
buffL       = 600
startSkip   = 30

# remove junk from start of file.
for i in range(startSkip):
    line    = sys.stdin.readline()

# Use the most recent line to determine how many columns (streams) there are.
lineInt     = [int(i) for i in line.split()]
lines       = len(lineInt)
databuff    = []
for i in range(lines):
    databuff.append( deque([0]*buffL))


# Pipe listenSomeLinks_v2.py output to this code.
# Run forever, adding lines as they are available.
counter = 0
while 1:
    line = sys.stdin.readline()
    if not line:
        continue
    while line[-1] != '\n':   # If line is incomplete, add to it.
        line += fin.readline()
    
    # Get the integers from the line string
    rss = [int(i) for i in line.split()]




#
# 2.2 Put RSS values in rssMat.  Each row is a link.
#
rssMat      = data[:,1::2].T

# link-channels are logical "links", that is, each channel is its own link.
linkchs, datalen = rssMat.shape

if plotOption:
    plt.figure(1)
    plt.plot(rssMat.T)
    plt.xlabel('Sample')
    plt.ylabel('RX Power (dBm)')
    plt.ylim(-100,-30)

#
# 3. Calculate the sampling period, window lengths, and DTFT matrix
#
run_duration = data[-1,-1] - data[0,-1]
sampling_period = run_duration / (datalen-1)  # seconds per sample, hard code if known.
sps          = 1.0 / sampling_period          # samples per second
N            = int(round(window_s * sps))     # samples per rate estimation window
incr_samp    = int(round(increment_s * sps))  # How many samples to move the window before re-estimating
HzRange      = np.arange(Min_RR, Max_RR, Delta_RR)  # Frequencies at which freq content is calc'ed.
freqRange    = (-2J * np.pi / sps)*HzRange    #normalized frequency
freqs        = len(freqRange)                 # How many frequencies at which DFT is calculated
# We'll multiply this by the data in the window to calculate PSD
expMat       = np.array([np.exp(freqRange*i) for i in np.arange(N)])

#
# 4. Compute the samples at which the RSS suddenly changes, which we call "breakpoints".
#
if breakptMeanSubtraction == 'common':
    wl          = 14    # window length, samples
    minStd      = 0.5   # avoid div-by-0 by having a minimum std deviation.
    absTSum, rmsTSum, twoGrpT, maxT = CalcTScore(rssMat, wl, minStd)        
    
    threshT = 1.5
    breakpts = set( np.argwhere(rmsTSum > threshT).flatten() )
    print "Number of breakpoints: " + str(len(breakpts))

elif breakptMeanSubtraction == 'wilcoxon':
    wl          = 14    # window length, samples
    absZSum, rmsZSum     = CalcWilcoxonScore(rssMat, wl)
    threshZ     = 1.15
    breakpts    = set( argwhere(rmsZSum > threshZ).flatten() )
    print "Number of breakpoints: " + str(len(breakpts))

elif breakptMeanSubtraction == 'cusum':
    wl          = 14    # window length, samples
    minStd      = 0.5   # avoid div-by-0 by having a minimum std deviation.
    k           = 1.2 # hand-chosen for sofa test for minStd = 0.5
    print 'k = ' + str(k)
    MC2 = CalcMultivariateCUSUMChart2(rssMat, wl, minStd, k)
    threshT = 0.0
    breakpts = set( argwhere(MC2 > threshT).flatten() )
    print "Number of breakpoints: " + str(len(breakpts))
    
elif breakptMeanSubtraction == 'link':
    wl          = 14    # window length, samples
    minStd      = 0.5   # avoid div-by-0 by having a minimum std deviation.
    absTSum, rmsTSum, twoGrpT, maxT = CalcTScore(rssMat, wl, minStd)
    absTwoGrpT  = abs(twoGrpT)
    threshT     = float(stats.scoreatpercentile(flatten(absTwoGrpT), 90.0))
    breakpts    = []
    for lc in range(linkchs):
        breakpts.append(set( argwhere(twoGrpT[lc, :] > threshT).flatten() ))
    print "Number of breakpoints: " + str(len(breakpts))

elif breakptMeanSubtraction == 'false':
    breakpts = []
    print "Number of breakpoints: " + str(len(breakpts))

# 4.1 Plot the values used to compute breakpoints, if desired.
if plotOption:
    if breakptMeanSubtraction == 'false':
        # do nothing
        print "Doing nothing"
    elif (breakptMeanSubtraction == 'common') or (breakptMeanSubtraction == 'link'):
        plt.figure(11)
        plt.clf()
        plt.plot(absTSum)
        plt.plot(rmsTSum)
        plt.plot(maxT/5.)
        plt.axis([0, len(rmsTSum), 0, max(maxT)/5.])
        plt.xticks(range(0,len(rmsTSum), 20))
        plt.grid()
    else:
        plt.figure(11)
        plt.clf()
        plt.plot(MC2)
        plt.grid()
        
            
#
# 5.  For each window, subtract mean of RSS, remove 127 values,
#     compute the frequency estimate and its error
#
start_ind_list = range(0, datalen-N, incr_samp)
fHat           = np.zeros(len(start_ind_list))
fError         = np.zeros(len(start_ind_list))
for ro, start_ind in enumerate(start_ind_list):
    print ro, start_ind

    # These commented functions do not deal with the 127-value problem.
    # Extract the RSS data from this window, and subtract the mean(s)
    #if breakptMeanSubtraction == 'link':
    #    meanRemoved = brokenSubtractMean(rssMat, start_ind, start_ind+N, breakpts)
    #elif (breakptMeanSubtraction == 'common') or (breakptMeanSubtraction == 'cusum') or (breakptMeanSubtraction == 'wilcoxon'):
    #    meanRemoved = brokenSubtractMeanCommon(rssMat, start_ind, start_ind+N, breakpts)
    #else: (rssMat, noMeastKey)

    # Remove the 127 values, and the mean.
    meanRemoved = removeMeanAndClean(rssMat[:,start_ind:(start_ind+N)], noMeastKey)

    # For this window of mean-removed RSS data, compute the frequency estimate
    FreqMS      = np.abs(np.dot(meanRemoved, expMat))**2.0 # Frequency mag sqr
    sumFreqMS   = np.mean(FreqMS, axis=0) * (4.0/N)
    fHatInd     = sumFreqMS.argmax()
    fHat[ro]    = HzRange[fHatInd]
    # Calculate performance of fHat estimate
    fError[ro]  = (fHat[ro] - const_breathing_rate)
    
    # Output the results.
    print 'Frequency estimate: ' + str(fHat[ro]*60.) + ' breaths per minute'
    print 'Frequency error: ' + str(fError[ro]*60.) + ' breaths per minute'

    # plot the results
    if plotOption:

        # Plot the Average PSD plot
        plt.figure(2)
        plt.clf()
        plt.plot(HzRange, sumFreqMS)
        plt.plot(HzRange[fHatInd], sumFreqMS[fHatInd],'ro')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized Average PSD')


        plt.figure(4)
        plt.clf()
        plt.plot(range(start_ind, start_ind + N), meanRemoved.T)
        plt.grid()
        plt.xlim(start_ind, start_ind+N-1)
        plt.xlabel('Sample Time Index', fontsize=18)
        plt.ylabel('Mean-Subtracted RSS (dB)', fontsize=18)

        raw_input()  # pause for user to hit Enter
        

# Compute average error
fHatRMSE        = np.sqrt(np.mean(fError**2))
fHatAvg         = np.mean(np.abs(fError))

#print 'fHatRMSE = ' + str(fHatRMSE)
print 'bpm RMS Err = ' + str(fHatRMSE * 60.0)
print 'bpm Avg Err = ' + str(fHatAvg * 60.0)
