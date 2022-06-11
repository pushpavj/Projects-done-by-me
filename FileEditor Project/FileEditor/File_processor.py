from search_file import find_text
from logger import worklogs

def process_file(file,find_string,replace_string):
    '''
    This module reads the input file and checks for the given string the record and writes the output
    file with the updated records

    '''
    worklogs.writelogs('File_processor', 'info', 'started with process_file ')
    try:

        input_file = open(file, 'rt+')
        l = len(input_file.readlines())

        in_rec = []
        out_rec = []
        input_file.seek(0)
    except Exception as e:
        worklogs.writelogs('File_processor', 'error', 'Error occurred while reading input file ')
        worklogs.writelogs('File_processor', 'exception', e)
    else:
        try:

            for i in range(l):
                read_rec = input_file.readline()
                in_rec.append(read_rec)

                write_rec = find_text(read_rec, find_string, replace_string)

                out_rec.append(write_rec)

            input_file.close()
        except Exception as e:
            worklogs.writelogs('File_processor', 'error', 'Error occurred while reading input records ')
            worklogs.writelogs('File_processor', 'exception', e)
        else:
            try:

                output_file = open(file, 'wt')
                for j in out_rec:
                    output_file.write(j)

                output_file.close()

                rcode=0
                rmessage2='File processed successfully'
            except Exception as e:
                worklogs.writelogs('File_processor', 'error', 'Error occurred while writing output file ')
                worklogs.writelogs('File_processor', 'exception', e)
            else:
                worklogs.writelogs('File_processor', 'INFO', 'File processed successfully' )

                return in_rec,out_rec,rcode,rmessage2
