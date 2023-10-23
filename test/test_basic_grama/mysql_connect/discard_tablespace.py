import glob
import os.path
import shutil
import traceback
from subprocess import Popen

import mysql.connector

DBHOST = "localhost"
DBUSER = "root"
DBPASS = "root"
DBNAME = "line4_10131"
port = "3306"

def create_database(db_name):
    sql="CREATE DATABASE  IF NOT EXISTS `"+db_name+"`  DEFAULT CHARACTER SET utf8 COLLATE utf8_unicode_ci  ;"


    mydb = mysql.connector.connect(
        host=DBHOST,
        user=DBUSER,
        password=DBPASS
    )

    cur = mydb.cursor()

    try:

        # 设置将执行的SQL语句
        cur.execute(sql)
        # 提交事务

    except Exception:
        print('【初始化失败（DB）】')
        # 打印错误信息
        print('    ', traceback.print_exc())


def discard_tablespace( table_name ):
    sql="ALTER TABLE  "+ table_name +"  DISCARD TABLESPACE;"
    print(sql)
    mydb = mysql.connector.connect(
        host=DBHOST,
        user=DBUSER,
        password=DBPASS,
           # 指定使用的数据库
           init_command = 'use  ' + DBNAME
    )

    cur = mydb.cursor()
    try:

        # 设置将执行的SQL语句
        cur.execute(sql)
        # 提交事务

    except Exception:
        print('【初始化失败（DB）】')
        # 打印错误信息
        print('    ', traceback.print_exc())


def import_tablespace( table_name ):
    sql="ALTER TABLE  "+ table_name +"  IMPORT  TABLESPACE;"
    mydb = mysql.connector.connect(
        host=DBHOST,
        user=DBUSER,
        password=DBPASS,
           # 指定使用的数据库
           init_command = 'use  ' + DBNAME
    )

    cur = mydb.cursor()
    try:

        # 设置将执行的SQL语句
        cur.execute(sql)
        # 提交事务

    except Exception:
        print('【初始化失败（DB）】')
        # 打印错误信息
        print('    ', traceback.print_exc())






datapath='D:/MySQL/MySQL Server 8.0/Data/line4_10131'

back_data='D:/MySQL/MySQL Server 8.0/Data/line4_10131_2'



ch_files = glob.glob(datapath + "/*")

for f in ch_files:
    base_name=os.path.basename(f)
    tab_name=base_name.replace(".ibd","")
    print(base_name)
    print(tab_name)

    # discard_tablespace(tab_name)
    # shutil.move(back_data+"/"+base_name,datapath)
    #
    import_tablespace(tab_name)




#             执行创建表
#              执行 清理表空间

