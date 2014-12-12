import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal
import scipy as sy

plt.ion()

def detectPeaks(q):
    peaks = signal.find_peaks_cwt(q, np.arange(1,20))
    result = [0]*len(q)
    for i in peaks:
        result[i] = 1
    return result

def main():

    result = []

    startskip    = 4  # Skip the first "startskip" rows
    endskip      = 4  # Skip the last "endskip" rows.

    data = np.genfromtxt("test.txt", skip_header=4, skip_footer=1)
    if data.shape[0] < (startskip + endskip + 2):
        sys.stderr.write('Error: data file is not long enough.\n')
        sys.exit(1)
    data         = data[startskip:-endskip, :]

    # "[sig: %i w: %i h: %i x: %i y: %i]"
    wMat      = data[:,1].T
    hMat      = data[:,2].T
    xMat      = data[:,3].T
    yMat      = data[:,4].T
    timeMat   = data[:,5].T

    #print wMat
    #print hMat
    #print xMat
    #print yMat
    #print timeMat

    #for i in range(len(timeMat)):
    #    print str(wMat[i]) + " " + str(hMat[i]) + " " + str(xMat[i]) + " " + str(yMat[i]) + " " + str(timeMat[i])

    result = detectPeaks(yMat)
    mins = 5
    print "Breath counts: " + str(sum(result)) + " Frequency: " + str(sum(result)/(float)(mins*60)) + " Hz"
    result = [150 if e == 1 else float('NaN') for e in result]

    plt.figure(1)
    plt.clf()

    plt.plot(timeMat, wMat, label="Width")
    plt.plot(timeMat, hMat, label="Height")
    plt.plot(timeMat, xMat, label="X")
    plt.plot(timeMat, yMat, label="Y")
    plt.plot(timeMat, result, linestyle='None', marker=r'+', label="Result")

    plt.xlabel('Time (ms)')
    plt.ylabel('Movement')
    plt.legend(loc='lower left')

    plt.grid()

    meanRemoved = yMat-np.mean(yMat)

    fft = sy.fftpack.rfft(meanRemoved)
    p = 20*np.log10(np.abs(fft))
    freq = sy.fftpack.rfftfreq(len(fft))

    normval = timeMat.shape[0]
    f = np.linspace(0.01, 50, 1000)
    pgram = signal.lombscargle(timeMat, meanRemoved, f)
    
    fig = plt.figure(2)
    plt.clf()

    fig.add_subplot(3, 1, 1)
    plt.plot(timeMat, yMat)
    plt.xlabel('time (ms)')
    plt.ylabel('raw')

    fig.add_subplot(3, 1, 2)
    plt.plot(freq, fft)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    
    #maxd = np.max(fft[np.where(freq>0)])
    maxd = np.max(fft)
    maxf = freq[np.where(fft == maxd)][0]
    plt.plot(maxf, maxd, '-x')

    print "Method 1 Breath freq: " + str(maxf) + " Hz"
    
    fig.add_subplot(3, 1, 3)
    normout = np.sqrt(4*(pgram/normval))
    plt.plot(f, normout)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')

    maxd = np.max(normout[np.where(f > 0.02)])
    maxf = f[np.where(normout == maxd)][0]
    plt.plot(maxf, maxd, '-x')

    print "Method 2 Breath freq: " + str(maxf) + " Hz"
    
    plt.grid()
    
    raw_input()

if __name__ == "__main__":
    main()
