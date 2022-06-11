import re
from logger import worklogs

def find_text(instring, findtext, replacetext):
    '''
    This module search for the given word in side the input rec and replaces it with the new word
    '''
    worklogs.writelogs('search_file', 'info', 'started with find_text ')
    try:

        a = ""
        pattern = re.compile(findtext)
        matches = pattern.finditer(instring)
        replacement_count = len(list(matches))
        if replacement_count > 0:
            a = pattern.sub(replacetext, instring, count=replacement_count)
        else:
            a = instring
    except Exception as e:
        worklogs.writelogs('search_file', 'error', 'Error occurred while searching file content for given text')
        worklogs.writelogs('search_file', 'exception', e)
    else:
        worklogs.writelogs('search_file', 'INFO', 'Searching file content for the given text is successful')
        return a
