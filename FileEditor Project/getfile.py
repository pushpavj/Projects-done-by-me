import os
from logger import worklogs

def searchfile(filename,filepath):
    '''
    This module combines the file name with its file path and checks for the availability of the file
    '''
    worklogs.writelogs('getfile', 'info', 'started with search file' )
    try:

        fullfile = os.path.join(filepath,filename)
        #fullfile=filepath+"\\"+filename
       
        print('filename=',filename)
        print('filepath=',filepath)
        print('fullname=',fullfile)
        print('cwd',os.getcwd())
        rcode = 0
        if os.path.exists(filepath):
             rcode=0
             rmessage='Let us see File present'
        else:
             rcode=100
             rmessage='File Not Found Please check the file path'
    except Exception as e:
        worklogs.writelogs('getfile', 'error', 'Error occurred while checking for file availability')
        worklogs.writelogs('getfile', 'exception', e)
    else:
        worklogs.writelogs('getfile', 'INFO', 'getfile module ran fine')

        return fullfile,rcode,rmessage
