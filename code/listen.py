import sys
import serial
import time
import rss

# Open port to listen node; 
serial_filename = rss.serialFileName()
sys.stderr.write('Using USB port file: ' + serial_filename + '\n')
ser = serial.Serial(serial_filename,38400)

# Continuously read any hex value on port and append to currentLine
currentLine = []
while(1):
    tempInt = ser.read().encode('hex')
    currentLine.append(tempInt)

    # If at end of line, print the line to the stdout with the current time
    if currentLine[-2:] == ['ef','be']:
        dataStr =  ", ".join([str(rss.hex2signedint(A)) for A in currentLine[:-2]])
        print "{0}, {1:f}".format(dataStr,time.time())
        currentLine = []