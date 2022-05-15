import pymongo
from MDB_logger import *

dblink="mongodb://localhost:27017/"
connect=pymongo.MongoClient(dblink)
"""
This module performs various operations on the mongodb such as, create data base,
create table, insert into tables, updates table records, deletes table records. 
"""

def create_DB(dbname):
    "This module creates the MongDB data base"
    writelogs('Access_mongoDB', 'info', 'Entered into create_DB')
    try:
        return connect[dbname]
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while creating DB')
        writelogs('create_DB', 'exception', e)

def create_table(tablename,dbname):
    "This module creates the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into create_table')
    try:
        return dbname[tablename]
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while creating table')
        writelogs('create_table', 'exception', e)

def single_insert(tablename,doc):
    "This module inserts record into the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into single_insert')
    try:
        tablename.insert_one(doc)
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while single insert')
        writelogs('single_insert', 'exception', e)


def bulk_insert(tablename,datalist):
    "This module inserts records into the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into bulk_insert')
    try:
        tablename.insert_many(datalist)
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while bulk insert')
        writelogs('bulk_insert', 'exception', e)


def single_find(tablename,query):
    "This module search record from the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into single_find')
    try:
        rec=tablename.find_one(query)
        return rec
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while single find')
        writelogs('single_find', 'exception', e)


def bulk_find(tablename,query):
    "This module search records from the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into bulk_find')
    try:
        rec=tablename.find(query)
        lst_rec=[]
        for i in rec:
            lst_rec.append(i)
        return lst_rec
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while bulk find')
        writelogs('bulk_find', 'exception', e)


def single_update(tablename,data1,data2):
    "This module updates record in the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into single_update')
    try:
        tablename.update_one(data1,data2)
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while single update')
        writelogs('single_update', 'exception', e)


def bulk_update(tablename,data1,data2):
    "This module update records in the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into bulk_update')
    try:
        tablename.update_many(data1,data2)
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while bulk update')
        writelogs('bulk_update', 'exception', e)

def single_delete(tablename,query1):
    "This module deletes record from the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into single_delete')
    try:
        tablename.delete_one(query1)
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while single delete')
        writelogs('single_delete', 'exception', e)

def bulk_delete(tablename,query2):
    "This module delete records from the MongDB collection name"
    writelogs('Access_mongoDB', 'info', 'Entered into bulk_delete')
    try:
        tablename.delete_many(query2)
    except Exception as e:
        writelogs('Access_mongoDB', 'error', 'Some error occurred while bulk update')
        writelogs('bulk_update', 'exception', e)

#******************End of Access_mondDB Module********************************




