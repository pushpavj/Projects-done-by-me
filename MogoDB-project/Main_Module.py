from Access_mongoDB import *
from MDB_logger import *
from Read_file import *


def singlefind():
       "This module calls single_find module to search DB"
       writelogs('Main Module', 'info', 'Entered into singlefind')
       try:
              query={"Initial atomic coordinate v":679005}
              data_found=single_find(collection,query)
              print(data_found)
       except Exception as e:
              writelogs('Main_Module', 'error', 'Some error occurred inside singlefind')
              writelogs('singlefind', 'exception', e)

def bulkfind():
       "This module calls bulk_find module to search DB"
       writelogs('Main Module', 'info', 'Entered into bulkfind')
       try:
              query={"Initial atomic coordinate v":{'$gte':679005}}
              data_found=bulk_find(collection,query)
              print(data_found)
       except Exception as e:
              writelogs('Main_Module', 'error', 'Some error occurred inside bulkfind')
              writelogs('bulkfind', 'exception', e)

def bulkupdation():
       "This module calls bulk_update module to update DB"
       writelogs('Main Module', 'info', 'Entered into bulkupdation')
       try:
              data1={"Initial atomic coordinate v":{'$lte':679005}}
              data2={"$set":{"Initial atomic coordinate w":5}}
              bulk_update(collection,data1,data2)
       except Exception as e:
              writelogs('Main_Module', 'error', 'Some error occurred inside bulkupdation')
              writelogs('bulkupdation', 'exception', e)

def singleupdation():
       "This module calls single_update module to update DB"
       writelogs('Main Module', 'info', 'Entered into singleupdation')
       try:
              data1={"Initial atomic coordinate v":679005}
              data2={"$set":{"Calculated atomic coordinates v":8}}
              single_update(collection,data1,data2)
       except Exception as e:
              writelogs('Main_Module', 'error', 'Some error occurred inside singleupdation')
              writelogs('singleupdation', 'exception', e)

def singledelete():
       "This module calls single_delete module to delete record in DB"
       writelogs('Main Module', 'info', 'Entered into singledelete')
       try:
              query={"Initial atomic coordinate v":679005}
              single_delete(collection,query)
       except Exception as e:
              writelogs('Main_Module', 'error', 'Some error occurred inside singledelete')
              writelogs('singledelete', 'exception', e)

def bulkdelete():
       "This module calls bulk_delete module to delete record in DB"
       writelogs('Main Module', 'info', 'Entered into bulkdelete')
       try:
              query={"Initial atomic coordinate v":{'$lte':679005}}
              bulk_delete(collection,query)
       except Exception as e:
              writelogs('Main_Module', 'error', 'Some error occurred inside bulkdelete')
              writelogs('bulkdelete', 'exception', e)

#Execution starts
db = create_DB('carbon_tubes')
db.dropDatabase

# Create a data base as carbon_tubes
try:
       db = create_DB('carbon_tubes')
except Exception as e:
       writelogs('Main_Module', 'error', 'Some error occurred while calling create_DB')
       writelogs('Main_Module', 'exception', e)

# create table as carbon_nanotubes
try:
       collection = create_table('carbon_nanatubes', db)
except Exception as e:
       writelogs('Main_Module', 'error', 'Some error occurred while calling create_table')
       writelogs('Main_Module', 'exception', e)

# Read the carbon_nanotubes.csv file
try:
       file = 'carbon_nanotubes.csv'
       data = readfile(file)
except Exception as e:
       writelogs('Main_Module', 'error', 'Some error occurred while calling readfile')
       writelogs('Main_Module', 'exception', e)

# format the data to be inserted into mongodb data base
try:
       dic_rec_list = format_rec(data)
except Exception as e:
       writelogs('Main_Module', 'error', 'Some error occurred while calling format_rec')
       writelogs('Main_Module', 'exception', e)

# Insert bulk record into Carbon_nanotubes collection
try:
       bulk_insert(collection, dic_rec_list)
except Exception as e:
       writelogs('Main_Module', 'error', 'Some error occurred while calling bulk_insert')
       writelogs('Main_Module', 'exception', e)

print('The MongoDB for Carbon nanotube is successfully created and data inserted\n')
options=0
try:
       while options != '7':
              options=input('''Select the number for your choice: 
                     1:find single record
                     2:find bulk record
                     3:single updation
                     4:bulk updation
                     5:single deletion
                     6:bulk deletion 
                     7:Exit options\n''')
              if options=='1':
                     singlefind()
              elif options=='2':
                     bulkfind()
              elif options=='3':
                     singleupdation()
              elif options=='4':
                     bulkupdation()
              elif options=='5':
                     singledelete()
              elif options=='6':
                     bulkdelete()
              else:
                     options=='7'
except Exception as e:
       writelogs('Main_Module', 'error', 'Some error occurred while chosing options')
       writelogs('Main_Module', 'exception', e)

#******************End of Main_Module Module********************************


