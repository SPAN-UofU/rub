#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

plt.ion()

plotOption   = False
fileOption   = False

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
    twoGrpZ       = np.zeros((linkchs, datalen-wl))

    for i in range(wl, datalen-wl):
        for ln in range(linkchs):
            twoGrpZ[ln,i], junk  = stats.ranksums(rssMat[ln, i-wl:i], rssMat[ln, i:i+wl])

    absZSum   = np.mean(np.abs(twoGrpZ), axis=0)
    rmsZSum   = np.sqrt(np.mean(np.abs(twoGrpZ)**2, axis=0))

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

# Return a 1-d array for any size input array
def flatten(inArray):

    outlen = np.prod(inArray.shape)
    return inArray.reshape(outlen,1)


# Calculate BPM
def calBPM():

    #
    # 1. Inputs to the script.
    #
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

    HzRange      = np.arange(Min_RR, Max_RR, Delta_RR)  # Frequencies at which freq content is calc'ed.

    # Parameters you may change:
    #   calcSkip:  Rerun the breathing calculation after this many data lines are read
    #   buffL:     buffer length, ie, how much data to plot.
    #   startSkip: A serial port seems to have a "memory" of several lines,
    #              which were saved from the previous experiment run.
    #              ** Must be greater than 0.
    plotSkip    = 1
    buffL       = 600 #30 sec
    avgL        = 100 #10 sec
    startSkip   = 300


    # Init output file
    if fileOption:
        fout = open('output.txt', 'w')

    # remove junk from start of file.
    line = []
    for i in range(startSkip):
        line    = sys.stdin.readline()

    # Use the most recent line to determine how many columns (streams) there are.
    #lineInt     = [int(i) for i in line.split()]
    #lines       = len(lineInt)
    #databuff    = []
    #for i in range(lines):
    #    databuff.append( deque([0]*buffL))

    # Use the most recent line to determine how many columns (streams) there are.
    lineInt = [float(i) for i in line.split()]
    columns = len(lineInt)
    streams = columns/2  # Every stream requires two columns (time, rss)
    rss     = [int(f) for f in lineInt[1::2]]  # take odd number columns
    #times   = [int(f) for f in lineInt[0:-1:2]] # take even number columns
    timeSec = lineInt[-1]

    # Init the figure.
    RSSBuffer   = []
    FBuffer = []
    TimeBuffer   = []
    linePlot    = []
    plt.clf()
    fig1 = plt.figure(num=1, figsize=(100, 100))
    ax1 = fig1.add_subplot(3, 1, 1)
    for n in range(streams):
        RSSBuffer.append( deque([rss[n]] * buffL))
        TimeBuffer.append( deque([timeSec] * buffL))
        l, = ax1.plot([0]*buffL, RSSBuffer[n], label=str(n))
        ax1.set_ylim((-95, -35))
        ax1.set_ylabel('Received Power (dBm)')
        ax1.set_xlabel('Measurement Time Ago (sec)')
        linePlot.append(l)

    #fig2 = plt.figure(2)
    ax2 = fig1.add_subplot(3, 1, 3)
    FBuffer = (deque([0] * buffL))
    l, = ax2.plot([0]*buffL, FBuffer)
    ax2.set_ylim((0, 30))
    ax2.set_ylabel('Breath Per Min (BPM)')
    ax2.set_xlabel('Measurement Time Ago (sec)')
    plotBuffer2 = l

    ax3 = fig1.add_subplot(3, 1, 2)
    FBuffer2 = (deque([0] * len(HzRange)))
    freq, = ax3.plot(HzRange, FBuffer2)
    maxFreq, = ax3.plot(0, 0,'ro')
    ax3.set_xlim((Min_RR, Max_RR))
    ax3.set_ylim((0, 30))
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Normalized Average PSD')
    

    # Init fHat and fError array
    fHat           = deque([])
    fError         = deque([])

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
            #ax1.set_axis([mintime, 0, -95, -35])
            ax1.set_xlim((mintime, 0))
            ax1.set_ylim((-95, -35))
            #plt.axis([mintime, 0, -70, -35])
            #plt.draw()

        #
        # 2.2 Put RSS values in rssMat.  Each row is a link.
        #
        rssMat      = np.array(RSSBuffer)

        # link-channels are logical "links", that is, each channel is its own link.
        linkchs, datalen = rssMat.shape

        if counter > buffL:

            #
            # 3. Calculate the sampling period, window lengths, and DTFT matrix
            #
            #run_duration = float(TimeBuffer[0][-1]) - float(TimeBuffer[0][0])
            #sampling_period = run_duration / (datalen-1)  # seconds per sample, hard code if known.
            sampling_period = np.max([float(TimeBuffer[i][-1]) - float(TimeBuffer[i][0]) for i in range(len(TimeBuffer))])/buffL
            sps          = 1.0 / sampling_period          # samples per second
            #N            = int(round(window_s * sps))     # samples per rate estimation window
            N            = buffL
            #incr_samp    = int(round(increment_s * sps))  # How many samples to move the window before re-estimating
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
                #print "Number of breakpoints: " + str(len(breakpts))

            elif breakptMeanSubtraction == 'wilcoxon':
                wl          = 14    # window length, samples
                absZSum, rmsZSum     = CalcWilcoxonScore(rssMat, wl)
                threshZ     = 1.15
                breakpts    = set( np.argwhere(rmsZSum > threshZ).flatten() )
                #print "Number of breakpoints: " + str(len(breakpts))

            elif breakptMeanSubtraction == 'cusum':
                wl          = 14    # window length, samples
                minStd      = 0.5   # avoid div-by-0 by having a minimum std deviation.
                k           = 1.2 # hand-chosen for sofa test for minStd = 0.5
                #print 'k = ' + str(k)
                MC2 = CalcMultivariateCUSUMChart2(rssMat, wl, minStd, k)
                threshT = 0.0
                breakpts = set( np.argwhere(MC2 > threshT).flatten() )
                #print "Number of breakpoints: " + str(len(breakpts))

            elif breakptMeanSubtraction == 'link':
                wl          = 14    # window length, samples
                minStd      = 0.5   # avoid div-by-0 by having a minimum std deviation.
                absTSum, rmsTSum, twoGrpT, maxT = CalcTScore(rssMat, wl, minStd)
                absTwoGrpT  = abs(twoGrpT)
                threshT     = float(stats.scoreatpercentile(flatten(absTwoGrpT), 90.0))
                breakpts    = []
                for lc in range(linkchs):
                    breakpts.append(set( np.argwhere(twoGrpT[lc, :] > threshT).flatten() ))
                #print "Number of breakpoints: " + str(len(breakpts))

            elif breakptMeanSubtraction == 'false':
                breakpts = []
                #print "Number of breakpoints: " + str(len(breakpts))

            """
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
            """

            #
            # 5.  For each window, subtract mean of RSS, remove 127 values,
            #     compute the frequency estimate and its error
            #

            # Remove the 127 values, and the mean.
            meanRemoved = removeMeanAndClean(rssMat, noMeastKey)

            # For this window of mean-removed RSS data, compute the frequency estimate
            FreqMS      = np.abs(np.dot(meanRemoved, expMat))**2.0 # Frequency mag sqr
            sumFreqMS   = np.mean(FreqMS, axis=0) * (4.0/N)
            fHatInd     = sumFreqMS.argmax()
            fHat.append(HzRange[fHatInd])
            if len(fHat) > avgL:
                fHat.popleft()
            # Calculate performance of fHat estimate
            fError      = np.append(fError, (fHat[-1] - const_breathing_rate))

            # Output the results.
            #print 'Frequency estimate: ' + str(fHat[-1]*60.) + ' breaths per minute'
            #print 'Frequency error: ' + str(fError[-1]*60.) + ' breaths per minute'

            """
            # plot the results
            if plotOption:

                # Plot the Average PSD plot
                plt.figure(2)
                plt.clf()
                plt.plot(HzRange, sumFreqMS)
                plt.plot(HzRange[fHatInd], sumFreqMS[fHatInd],'ro')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Normalized Average PSD')

                plt.figure(3)
                plt.clf()
                plt.plot(range(start_ind, start_ind + N), meanRemoved.T)
                plt.grid()
                plt.xlim(start_ind, start_ind+N-1)
                plt.xlabel('Sample Time Index', fontsize=18)
                plt.ylabel('Mean-Subtracted RSS (dB)', fontsize=18)

                plt.draw()
                #raw_input()  # pause for user to hit Enter
            """

            # Compute average error
            fHatRMSE        = np.sqrt(np.mean(fError**2))
            fHatAvg         = np.mean(np.abs(fError))

            #print 'fHatRMSE = ' + str(fHatRMSE)
            #print 'bpm Avg = ' + str(np.average(fHat)*60.0)
            #print 'bpm RMS Err = ' + str(fHatRMSE * 60.0)
            #print 'bpm Avg Err = ' + str(fHatAvg * 60.0)

            if fileOption:
                # <Second> <frequency estimate> <frequency error> <bpm avg> <RMSE> <Avg Err>
                fout.write(str(timeSec) + " " + \
                    str(fHat[-1]*60.) + " " + \
                    str(fError[-1]*60.) + " " + \
                    str(np.average(fHat)*60.0) + " " + \
                    str(fHatRMSE * 60.0) + " " + \
                    str(fHatAvg * 60.0) + "\n")
            oldF  = FBuffer.popleft()
            FBuffer.append(fHat[-1]*60.0)

            # Update plot
            plotBuffer2.set_ydata(FBuffer)
            plotBuffer2.set_xdata(relTime)
            ax2.set_xlim((mintime, 0))
            ax2.set_ylim((0, 30))

            freq.set_ydata(sumFreqMS)
            maxFreq.set_xdata(HzRange[fHatInd])
            maxFreq.set_ydata(sumFreqMS[fHatInd])

            plt.draw()



#####################################################################
#
#  Main
#
#####################################################################

def main():
    calBPM()

if __name__ == "__main__":
    main()
