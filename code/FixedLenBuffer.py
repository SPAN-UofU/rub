# ########################################
# Code to provide a fixed-length buffer data type
class FixedLenBuffer:
    def __init__(self, initlist):
        self.frontInd = 0
        self.data     = initlist
        self.len      = len(initlist)
    
    def list(self):
        oldest = self.frontInd+1
        return self.data[oldest:] + self.data[:oldest]
    
    # Append also deletes the oldest item
    def append(self, newItem):
        self.frontInd += 1
        if self.frontInd >= self.len:
            self.frontInd = 0
        self.data[self.frontInd] = newItem
    
    # Returns the "front" item
    def mostRecent(self):
        return self.data[self.frontInd]
    
    # Returns the N items most recently appended
    def mostRecentN(self,N):
        return [self.data[(self.frontInd-i)%self.len] for i in range(N-1,-1,-1)]
    
    # Returns the variance of the data
    def var(self):
        return array(self.data).var()

    def dataWin(self, start, end):
        return [self.data[(self.frontInd-i)%self.len] for i in range(start+1,stop+1)]
# ########################################
