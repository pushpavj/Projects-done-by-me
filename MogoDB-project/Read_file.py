import csv
from MDB_logger import *

'''
 This module performs reading CSV file and formats the CSV file records so that
 can be inserted properly into to the MONGO DB data base
'''

def readfile(filename):
    "This module reads the csv file"
    writelogs('Read_file', 'info', 'Entered into readfile')

    try:

        lst=[]
        with open('carbon_nanotubes.csv','r') as file:
            file_csv=csv.reader(file,delimiter='\n')
            for i in file_csv:
                a=i[0].replace(';', ',')
                lst.append(a.replace("'", ""))
            return lst
    except Exception as e:
        writelogs('Read_file', 'error', 'Some error occurred while reading csv file')
        writelogs('readfile', 'exception', e)


def format_rec(lst):
    "This module formats the csv file record to be inserted into MongoDB"
    writelogs('Read_file', 'info', 'Entered into format_rec')

    try:
        cols = lst[0]
        colname = cols.split(',')
        lst2 = []

        for i in range(1, len(lst)):
            lst3 = []
            rec = lst[i].split(',')
            for j in rec:
                lst3.append(int(j))
                dic = dict(zip(colname, lst3))
                lst2.append(dic)
        return lst2
    except Exception as e:
        writelogs('Read_file', 'error', 'Some error occurred while formatting csv file')
        writelogs('format_rec', 'exception', e)


#******************End of Read_file Module********************************