from serial import *
import pygame
import time
import datetime
import sys

file_step = 'step_beep.mp3'
file_stop = 'stop_beep.mp3'

pygame.init()
pygame.mixer.init()

# 30 BPM
# 60 step and stop per minute
# 60 sec / 60 step = 1 step per sec
delay = 2 #second

while True:
    pygame.mixer.music.load(file_step)
    pygame.mixer.music.play()
    
    time.sleep(delay)
    
    pygame.mixer.music.load(file_step)
    pygame.mixer.music.play()

    time.sleep(delay)
