import sys
import platform
import glob

def hex2signedint(he):
    # Convert from hexidecimal 2's complement to signed 8 bit integer
    return (int(he,16) + 2**7) % 2**8 - 2**7

def prevChannel(channelList, ch_now):
    if (channelList.count(ch_now) > 0):
        i = channelList.index(ch_now)
        rval = channelList[(i-1) % len(channelList)]
    else:
        rval = -1  # Key for bad ch_now input
    return rval

# USER: The following serial "file name" changes depending on your operating
#       system, and what name is assigned to the serial port when your listen
#       node is plugged in.
def serialFileName():    
    system_name = platform.system()
    #
    # LINUX USERS
    if system_name == 'Linux':
        # Automatically grab the USB filename (since the number after /dev/ttyACM may vary)
        usb_file_list = glob.glob('/dev/ttyACM*')
        if len(usb_file_list) > 0:
            serial_filename =  usb_file_list[0]  
        else:
            sys.stderr.write('Error: No Listen node plugged in?\n')
        serial_filename = '0'
    #
    # WINDOWS USERS: Change 'COM#' to match what the system calls your USB port.
    elif system_name == 'Windows':
        serial_filename = 'COM3'
    #
    # MAC USERS
    else:  # 'Darwin' indicates MAC OS X
        # Automatically grab the USB filename (since the number after /dev/tty.usb may vary)
        usb_file_list = glob.glob('/dev/tty.usb*')
        if len(usb_file_list) > 0:
            serial_filename =  usb_file_list[0]  
        else:
            sys.stderr.write('Error: No Listen node plugged in?\n')
    #
    return serial_filename
    
