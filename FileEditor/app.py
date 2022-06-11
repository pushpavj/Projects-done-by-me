from flask import Flask, render_template, request, jsonify
from logger import worklogs
from flask_cors import CORS, cross_origin



from File_processor import process_file
import getfile
'''
This is the main app module for File Eiditor app
This app edits the text file given by the user by finding and replacing the user specified string/text
This module uses Flask as API

'''

worklogs.writelogs('Start1', 'info', 'started with app module')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) # To render Homepage
@cross_origin()
def home_page():
    '''
    This function renders the html template for user to provide the text file details
    '''

    worklogs.writelogs('start2', 'info', 'started with home page ')

    return render_template('index.html')

@app.route('/file', methods=['POST'])  # This will be called from UI
@cross_origin()
def file_operation():
    '''
    This function processes the file details and renders the html template with the result of
    file process.
    '''

    worklogs.writelogs('start3', 'info', 'started with file_operation ')

    if (request.method=='POST'):
        try:

            filename=request.form['filename']
            filepath = request.form['filepath']
            find_string=request.form['find_string']
            replace_string=request.form['Replace_string']
            rcode = 0
            ''' 
            Call getfile to get the full file name check the presence of file in the given
            Direcotory
            '''
            file,rcode,rmessage=getfile.searchfile(filename,filepath)

            if rcode > 0 :
                result='ERROR: The given File name with path is ' + str(file) + '\n'
                result2=rmessage
                rcode=1
            else:
                result = 'Success: The filename with path is  ' + str(file) + '\n'
                result2=rmessage

            if rcode == 0:
               '''
               Call process_file to process the user provided file
               '''
               inrec,outrec,rcode,rmessage2=process_file(file,find_string,replace_string)

               if rcode > 0:
                   result3='File Could not be processed, something went wrong'
                   result4=rmessage2
                   rcode=2
               else:
                   result3='File successfully processed'
                   result4='In put file records were                 '
                   result5=inrec
                   result6='Successfully replaced with            '
                   result7=outrec


            if rcode==1:
                result3,result4, result5,result6,result7='','','','',''
            elif rcode==2:
                result5, result6, result7 = '','',''

        except Exception as e:
            worklogs.writelogs('app', 'error', 'Error occurred while performing file editing operations')
            worklogs.writelogs('app', 'exception', e)
        else:
            worklogs.writelogs('app', 'info', 'File Editing is successful')

            return render_template('results.html', result=result, result2=result2,
                               result3=result3, result4=result4, result5=result5,
                               result6=result6, result7=result7)


if __name__ == '__main__':
    app.run(debug=True)
