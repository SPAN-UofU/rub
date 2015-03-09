# first pattern definition
!part1 = [
30,2,4,1,1
]

# second pattern definition
!part2 = [
40,2,4,1,1
]

# third pattern definition
!part3 = [
35,2,4,1,1
]

#play first pattern
# 30 BPM 30 times = 1 minute
# * 5 
# First minute calibration
@part1,30
# Next 5 minutes
@part1,150

#play second pattern
# 40 bpm 40 times = 1 minute
# * 5
@part2,200

#play third pattern
@part3,175