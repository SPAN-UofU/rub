#! /usr/bin/env python

# This script reads packet data from the listen node through the serial port
# and prints one data line for each tx, rx, ch measurement, and includes
# the time the measurement was recorded, and the measured RSS.

#
# Version History:
#
# Version 1.0:  Initial Release
#
# Version 1.1:  26 Sept 2012.
#   Fixes problems that occur when packets are dropped:
#   a) Changes the way the "ch", the channel the RSS was measured on,
#      is calculated.  If rxId > txId, then "ch" is the same as the current
#      channel.  Otherwise ch is the previous channel in channelList.
#      This is always correct.  The old way could be in error if a packet
#      was dropped by the listen node.
#   b) Does not try to prevent output of a 127 "no measurement key".  This
#      number is returned by the nodes themselves when the don't measure
#      the RSS on a link.  Downstream algorithms can use that information
#      differently, depending on the application.  Previous code copied the
#      most recently measured value for that link into the current line.
#   c) Changes when a line is output.  Previous code output every cycle, that
#      is, at the lowest channel for the lowest node id.  Now, we simply
#      keep track of which links have received a RSS measurement during this
#      line.  The line is output whenever any link sees a 2nd measurement.
#      This ensures that no data is overwritten in the case that the listen
#      node hears few packets that arrive seemingly out of order.
#
# Version 1.2: 1 Sept 2014.
#   a) Adds a column with the local PC clock time.  There is only one 
#      floating point clock time per row, we use the maximum time of the
#      measurements on all the channels.
#   b) Common functions are now in python library rss.py.
#      

import time
import rss
import serial
import sys

# What channels are measured by your nodes, in order of their measurement?
# USER:  SET THIS TO THE CHANNELS IN YOUR CHANNEL GROUP
channelList   = [26, 11, 16, 21]
#channelList   = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

# What node numbers are yours, that you want to see output to the file.
# USER:  SET THIS TO THE NODE IDS ASSIGNED TO YOU.  DO NOT INCLUDE THE LISTEN NODE NUMBER
nodeList      = [1,2]

# A link is a (transmitter id, receiver id, channel number) combination.
#    11 <= channel number <= 26, but is limited to those in "channelList"
#    transmitter and receiver id are positive integers in nodeList
# Examples:
#linksToPlot = [ (5,6,22), (6,5,22), (5,6,26), (6,5,26)]
#linksToPlot = [ (6,5,14), (6,5,18), (6,5,22), (6,5,26)]
linksToPlot = [ (1,2,11), (1,2,16), (1,2,21), (1,2,26)]
#linksToPlot = [ (5,6,14), (5,6,18), (5,6,22), (5,6,26)]
#linksToPlot = [ (1,2,11), (2,1,11)]
#linksToPlot = [ (1,2,11), (1,2,12), (1,2,13), (1,2,14), (1,2,15), (1,2,16), (1,2,17), (1,2,18), (1,2,19), (1,2,20), (1,2,21), (1,2,22), (1,2,23), (1,2,24), (1,2,25), (1,2,26)]

# Open the serial port at 38400 bits/sec.
serial_filename = rss.serialFileName()
sys.stderr.write('Using USB port file: ' + serial_filename + '\n')
ser = serial.Serial(serial_filename,38400)

# How many nodes the sensors have as the max # nodes (what # they're programmed with)
maxNodes      = 2

# Parameters that are due to our implementation of the listen node.
numNodes      = len(nodeList)
rssIndex      = 3
string_length = maxNodes + 7
suffix        = ['ef','be']  # "0xBEEF"

# Initialize data, output file
nodeSet       = set(nodeList)
channelSet    = set(channelList)
currentLine   = []  # Init serial data buffer "currentLine" as empty.
# Higher than any possible RSS value.  Match the embedded code on device.
noMeasurementKey = 127
currentLinkRSS  = [noMeasurementKey] * len(linksToPlot)
currentLinkTime = [0] * len(linksToPlot)
currentLinkTimeSec = [0.0] * len(linksToPlot)
prevTimeForTxId = [0] * (maxNodes+1)
prevTimeSecForTxId = [0.0] * (maxNodes+1)

# Run forever, adding one integer at a time from the serial port,
#   whenever an integer is available.
while(1):
    tempInt = ser.read().encode('hex')
    currentLine.append(tempInt)

    # Whenever the end-of-line sequence is read, operate on the "packet" of data.
    if currentLine[-2:] == suffix:
        if len(currentLine) != string_length:
            sys.stderr.write('packet corrupted - wrong string length\n')
            del currentLine[:]
	    continue
	#sys.stderr.write(' '.join(currentLine)  + '\n')
	currentLineInt = [int(x, 16) for x in currentLine]
	rxId = currentLineInt[2]
	currentCh = currentLineInt[-4]
	if (rxId not in nodeSet):
	    sys.stderr.write('rxId of ' + rxId + ' not in nodeSet\n')
	    del currentLine[:]
	    continue
	if (currentCh not in channelSet):
	    sys.stderr.write('Channel ' + currentCh + ' not in channelSet\n')
	    del currentLine[:]
	    continue
	timeStamp = 256*currentLineInt[1] + currentLineInt[0]
	timeStampSec = time.time()

	# The most recent transmission of node rxId is the time stamp reported
	# on this line.
	prevTimeForTxId[rxId] = timeStamp
	prevTimeSecForTxId[rxId] = timeStampSec

	# Each line in the serial data has RSS values for multiple txids.
	# Output one line per txid, rxid, ch combo.
	for txId in nodeList:
	    # If the rxId is after the txId, then no problem -- currentCh
	    # is also the channel that node txId was transmitting on when
	    # node rxId made the measurement, because nodes transmit on a
	    # channel in increasing order.
	    if rxId > txId:
	        ch = currentCh
	    else:
                ch = rss.prevChannel(channelList, currentCh)

	    if (linksToPlot.count((txId, rxId, ch)) > 0):
                i = linksToPlot.index((txId, rxId, ch))
	        # If the RSS has already been recorded for this link on
		# this "line", then output the line first, and then restart
		# with a new line.
		if currentLinkRSS[i] < noMeasurementKey:
		    # Output currentLinkRSS & currentLinkTime vectors
		    s = str(currentLinkTime[0]) + ' ' + str(currentLinkRSS[0])
		    for n in range(1,len(linksToPlot)):
		        s += ' ' + str(currentLinkTime[n]) + ' ' + str(currentLinkRSS[n])
		    
		    timeSecStr = ' {:.5f}'.format(max(currentLinkTimeSec))
		    sys.stdout.write(s + timeSecStr + '\n')
		    sys.stdout.flush()
		    # Restart with a new line by resetting currentLinkRSS and
		    # currentLinkTime
		    currentLinkRSS  = [noMeasurementKey] * len(linksToPlot)
		    currentLinkTime = [0] * len(linksToPlot)
		    currentLinkTimeSec = [0.0] * len(linksToPlot)

		# Store the RSS & time it was recorded.
		currentLinkRSS[i] = rss.hex2signedint(currentLine[rssIndex+txId-1])
		currentLinkTime[i] = prevTimeForTxId[txId]
		currentLinkTimeSec[i] = prevTimeSecForTxId[txId]

	# Remove serial data from the buffer.
	currentLine = []
