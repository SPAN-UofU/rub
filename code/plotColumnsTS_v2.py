#! /usr/bin/env python

#
# Version History:
#
# Version 1.0:  Initial Release
#
# Version 1.1:  27 Sept 2012.
#   Change: This version won't delete (pop) data unless it is going to enter
#           in a new piece of data.
#

import sys
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Parameters you may change:
#   plotSkip:  Refresh the plot after this many data lines are read
#   buffL:     buffer length, ie, how much data to plot.
#   startSkip: A serial port seems to have a "memory" of several lines,
#              which were saved from the previous experiment run.
#              ** Must be greater than 0.
plotSkip    = 1
buffL       = 80
startSkip   = 30

# remove junk from start of file.
for i in range(startSkip):
  line = sys.stdin.readline()

# Use the most recent line to determine how many columns (streams) there are.
lineInt = [float(i) for i in line.split()]
columns = len(lineInt)
streams = columns/2  # Every stream requires two columns (time, rss)
rss     = [int(f) for f in lineInt[1::2]]  # take odd number columns
#times   = [int(f) for f in lineInt[0:-1:2]] # take even number columns
timeSec = lineInt[-1]

# Init the figure.
plt.ion()
RSSBuffer   = []
TimeBuffer   = []
linePlot    = []
plt.cla()
for n in range(streams):
    RSSBuffer.append( deque([rss[n]] * buffL))
    TimeBuffer.append( deque([timeSec] * buffL))
    l, = plt.plot([0]*buffL, RSSBuffer[n], label=str(n))
    plt.ylim((-95, -25))
    plt.ylabel('Received Power (dBm)')
    plt.xlabel('Measurement Time Ago (sec)')
    linePlot.append(l)

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
        plt.axis([mintime, 0, -95, -35])
        #plt.axis([mintime, 0, -70, -35])
        plt.draw()
