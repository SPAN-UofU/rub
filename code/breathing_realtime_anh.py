#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
import collections, itertools
import scipy.stats as stats
import datetime

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

plt.ion()

plotOption   = None
fileOption   = True


# Plot highest link lines
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

def CalcTScoreRT(meanBuffer, varBuffer, wl, minStd):
    links         = len(meanBuffer)
    twoGrpStd     = [np.sqrt(varBuffer[i][-1] + varBuffer[i][0])/np.sqrt(wl) for i in range(links)]
    twoGrpStdwMin = np.array([max(minStd, v) for v in twoGrpStd])
    twoGrpT       = [(meanBuffer[i][-1] - meanBuffer[i][0])/ twoGrpStdwMin[i] for i in range(links)]
    rmsTSum       = np.sqrt(np.mean(np.abs(twoGrpT)**2))
    return rmsTSum

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

# Return a 1-d array for any size input array
def flatten(inArray):

    outlen = np.prod(inArray.shape)
    return inArray.reshape(outlen,1)

# Return Exponential Matrix
def calcExpMatrix(sampling_period, N, HzRange):
    sps          = 1.0 / sampling_period          # samples per second
    #N            = int(round(window_s * sps))     # samples per rate estimation window
    #incr_samp    = int(round(increment_s * sps))  # How many samples to move the window before re-estimating
    freqRange    = (-2J * np.pi / sps)*HzRange    #normalized frequency
    freqs        = len(freqRange)                 # How many frequencies at which DFT is calculated
    # We'll multiply this by the data in the window to calculate PSD
    expMat       = np.array([np.exp(freqRange*i) for i in np.arange(N)])
    return expMat


# Calculate BPM
def calBPM():

    # Inputs to the script.
    # if the breathing rate was constant throughout the experiment.
    #const_breathing_rate = 15.0/60.0  # true breaths per second
    noMeastKey   = 127

    # Breathing estimator parameters and options
    breakptMeanSubtraction = 'false'  # 'common', 'link', or 'false' or 'cusum' or 'wilcoxon'
    window_s     = 30       # Duration of time window, seconds
    increment_s  = 5        # seconds of window overlap, seconds
    Min_RR       = 0.1      # Minimum breathing rate, Hz
    Delta_RR     = 0.002    # Granularity of spectrum estimate
    Max_RR       = 0.4      # Maximum breathing rate, Hz

    HzRange      = np.arange(Min_RR, Max_RR, Delta_RR)  # Frequencies at which freq content is calc'ed.

    # Parameters you may change:
    #   calcSkip:  Rerun the breathing calculation after this many data lines are read
    #   buffL:     buffer length, ie, how much data to plot.
    #   startSkip: A serial port seems to have a "memory" of several lines,
    #              which were saved from the previous experiment run.
    #              ** Must be greater than 0.
    plotSkip    = 5
    buffL       = 600 #30 sec
    avgL        = 100 #10 sec
    startSkip   = 1

    # Init output file
    if fileOption:
        fout = open('../output.txt', 'w')

    # remove junk from start of file.
    line = []
    for i in range(startSkip):
        line    = sys.stdin.readline()

    # Use the most recent line to determine how many columns (streams) there are.
    lineInt = [float(i) for i in line.split()]
    columns = len(lineInt)
    streams = columns/2  # Every stream requires two columns (time, rss)
    rss     = [int(f) for f in lineInt[1::2]]  # take odd number columns
    timeSec = lineInt[-1]

    # Init the figure.
    RSSBuffer   = []
    FBuffer     = []
    TimeBuffer  = []
    linePlot    = []
    linePlot2   = []
    varBuffer   = []
    meanBuffer  = []
    MSRBuffer   = [] # Mean subtracted rss buffer
    wl = 15
    plt.clf()
    fig1 = plt.figure(num=1)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    ax1 = fig1.add_subplot(4, 1, 1)
    ax4 = fig1.add_subplot(4, 1, 2)
    for n in range(streams):
        RSSBuffer.append( deque([rss[n]] * buffL))
        TimeBuffer.append( deque([timeSec] * buffL))
        
        varBuffer.append( deque([0] * (wl + 1)))
        meanBuffer.append( deque([0] * (wl + 1)))
        MSRBuffer.append( deque([rss[n]] * buffL))
        
        l, = ax1.plot([0]*buffL, RSSBuffer[n], label=str(n))
        ax1.set_ylim((-95, -35))
        ax1.set_ylabel('Received Power (dBm)')
        ax1.set_xlabel('Measurement Time Ago (sec)')
        linePlot.append(l)
        
        l, = ax4.plot([0]*buffL, MSRBuffer[n], label=str(n))
        ax4.set_ylim((-10, 10))
        ax4.set_ylabel('Mean-Subtracted RSS (dB)')
        ax4.set_xlabel('Measurement Time Ago (sec)')
        linePlot2.append(l)

    ax2 = fig1.add_subplot(4, 1, 4)
    FBuffer = (deque([0] * buffL))
    l, = ax2.plot([0]*buffL, FBuffer)
    ax2.set_ylim((0, 40))
    ax2.set_ylabel('Breath per Minute (BPM)')
    ax2.set_xlabel('Measurement Time Ago (sec)')
    plotBuffer2 = l

    ax3 = fig1.add_subplot(4, 1, 3)
    FBuffer2 = (deque([0] * len(HzRange)))
    freq, = ax3.plot(HzRange, FBuffer2)
    maxFreq, = ax3.plot(0, 0,'ro')
    ax3.set_xlim((Min_RR, Max_RR))
    ax3.set_ylim((0, 15))
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Normalized Average PSD')

    # Init fHat and fError array
    fHat           = deque([])
    #fError         = deque([])

    # Pipe listenSomeLinks_v2.py output to this code.
    # Run forever, adding lines as they are available.
    counter = 0
    currentBP = -1
    previousBP = -1
    while 1:
        try:
            line = sys.stdin.readline()
            if not line:
                continue
            while line[-1] != '\n':   # If line is incomplete, add to it.
                line += fin.readline()

            # Get the integers from the line string
            data = [float(i) for i in line.split()]
            rss  = [int(f) for f in data[1::2]]  # take odd number columns
            #times = data[0:-1:2] # take even number columns except for final column
            timeSec = data[-1]

            # Append the queue for each stream with the newest RSS and Timestamps
            for i in range(streams):
                # data > -10 indicates no data measured.  Don't include a new value.
                if (rss[i] < -10):
                    oldRSS = RSSBuffer[i].popleft()
                    oldTS  = TimeBuffer[i].popleft()
                    RSSBuffer[i].append(rss[i])
                    TimeBuffer[i].append(timeSec)
                    
                    # calculate the mean, variance, std for wl windows
                    meanBuffer[i].popleft()
                    meanBuffer[i].append(np.mean(list(RSSBuffer[i])[-wl:]))
                    
                    varBuffer[i].popleft()
                    varBuffer[i].append(np.var(list(RSSBuffer[i])[-wl:]))
                    
                    MSRBuffer[i].popleft()
                    MSRBuffer[i].append(0)
                              

            # Every plotSkip rows, redraw the plot.  Time stamps, relative to the
            #   maximum time stamp, are on the x-axis.
            counter += 1
            if np.mod(counter, plotSkip) == 0:
                mintime = 0
                for i in range(streams):
                    linePlot[i].set_ydata(RSSBuffer[i])
                    relTime = np.array(TimeBuffer[i]) - max(TimeBuffer[i])
                    linePlot[i].set_xdata(relTime)
                    mintime = min(min(relTime), mintime)
                ax1.set_xlim((mintime, 0))
                ax1.set_ylim((-95, -35))

            # Put RSS values in rssMat.  Each row is a link.
            rssMat      = np.array(RSSBuffer)

            # link-channels are logical "links", that is, each channel is its own link.
            linkchs, datalen = rssMat.shape

            # Breakpoints Removal
            currentBP -= 1
            if currentBP < -buffL:
                currentBP = -buffL

            previousBP -= 1
            if previousBP < -buffL:
                previousBP = -buffL


            if counter > wl:
                # Call calcTScoreRT
                minStd      = 0.5   # avoid div-by-0 by having a minimum std deviation.
                threshT     = 1.5
                rmsTSum     = CalcTScoreRT(meanBuffer, varBuffer, wl, minStd)
                if rmsTSum > threshT:
                    # set current breakpoint
                    previousBP = currentBP
                    currentBP = -wl
                    
                    # calculate the new mean from previous breakpoint until now
                    for i in range(linkchs):
                        MSRBuffer[i] = deque(
                            list(MSRBuffer[i])[:previousBP] +
                            list(list(RSSBuffer[i])[previousBP:currentBP+1] - np.mean(list(RSSBuffer[i])[previousBP:currentBP+1])) +
                            list(MSRBuffer[i])[currentBP+1:]
                            )

            if counter == buffL+1:
                # Calculate the sampling period, window lengths, and DTFT matrix
                #run_duration = float(TimeBuffer[0][-1]) - float(TimeBuffer[0][0])
                #sampling_period = run_duration / (datalen-1)  # seconds per sample, hard code if known.
                sampling_period = np.average([  np.median([float(TimeBuffer[i][j]) - float(TimeBuffer[i][j-1]) for j in range(1,buffL)]) for i in range(len(TimeBuffer))])
                expMat = calcExpMatrix(sampling_period, buffL, HzRange)

            elif counter > buffL+1:
                # New sampling periods
                sampling_period_new = np.average([  np.median([float(TimeBuffer[i][j]) - float(TimeBuffer[i][j-1]) for j in range(1,buffL)]) for i in range(len(TimeBuffer))])
                if (sampling_period_new - sampling_period)/sampling_period > 0.0001:
                    expMat = calcExpMatrix(sampling_period_new, buffL, HzRange)

                for i in range(linkchs):
                    MSRBuffer[i] = deque(
                        list(MSRBuffer[i])[:currentBP+1] +
                        list(list(RSSBuffer[i])[currentBP+1:] - np.mean(list(RSSBuffer[i])[currentBP+1:]))
                        )

                # Remove the 127 values, and the mean.
                #meanRemoved = removeMeanAndClean(rssMat, noMeastKey)

                # For this window of mean-removed RSS data, compute the frequency estimate
                FreqMS      = np.abs(np.dot(MSRBuffer, expMat))**2.0 # Frequency mag sqr
                N           = buffL
                sumFreqMS   = np.mean(FreqMS, axis=0) * (4.0/N)
                fHatInd     = sumFreqMS.argmax()
                fHat.append(HzRange[fHatInd])
                if len(fHat) > avgL:
                    fHat.popleft()
                oldF  = FBuffer.popleft()
                FBuffer.append(fHat[-1]*60.0)

                # Every plotSkip rows, redraw the plot.  Time stamps, relative to the
                #   maximum time stamp, are on the x-axis.
                counter += 1
                if np.mod(counter, plotSkip) == 0:
                    mintime = 0
                    for i in range(streams):
                        linePlot2[i].set_ydata(MSRBuffer[i])
                        relTime = np.array(TimeBuffer[i]) - max(TimeBuffer[i])
                        linePlot2[i].set_xdata(relTime)
                        mintime = min(min(relTime), mintime)
                    ax4.set_xlim((mintime, 0))
                    ax4.set_ylim((-10, 10))
                    
                    # Update plot
                    plotBuffer2.set_ydata(FBuffer)
                    plotBuffer2.set_xdata(relTime)
                    ax2.set_xlim((mintime, 0))
                    ax2.set_ylim((0, 40))

                    freq.set_ydata(sumFreqMS)
                    maxFreq.set_xdata(HzRange[fHatInd])
                    maxFreq.set_ydata(sumFreqMS[fHatInd])

                    plt.draw()

                # Calculate performance of fHat estimate for defined constant rate
                #fError      = np.append(fError, (fHat[-1] - const_breathing_rate))

                # Compute average error
                #fHatRMSE        = np.sqrt(np.mean(fError**2))
                #fHatAvg         = np.mean(np.abs(fError))

                if fileOption:
                    # <Second> <frequency estimate> <frequency error> <bpm avg> <RMSE> <Avg Err>
                    """
                    fout.write(str(timeSec) + " " + \
                        str(fHat[-1]*60.) + " " + \
                        str(fError[-1]*60.) + " " + \
                        str(np.average(fHat)*60.0) + " " + \
                        str(fHatRMSE * 60.0) + " " + \
                        str(fHatAvg * 60.0) + "\n")
                    """
                    fout.write(str(timeSec) + " " + \
                        str(fHat[-1]*60.) + "\n")
    
        except (KeyboardInterrupt, SystemExit):
            plt.close(fig1)
            quit()



#####################################################################
#
#  Main
#
#####################################################################

def main():
    calBPM()

if __name__ == "__main__":
    main()
