import glob
import os.path
import traceback
from subprocess import Popen

import mysql.connector

DBHOST = "localhost"
DBUSER = "root"
DBPASS = "root"
DBNAME = "hih"
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


def import_tablespace( table_name ):
    sql="ALTER TABLE  "+ table_name +"  IMPORT  TABLESPACE;"
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






datapath='D:\MySQL\MySQL Server 8.0\Data'

files= [os.path.join(datapath,file) for file in os.listdir(datapath)]



for file in files:
    if  os.path.isdir(file)  and os.path.basename(file).find("inno")< 0   :


        os.chdir(file)
        database = os.path.basename(file)

        print("database="+database)

        os.rename(database, database+"_tmp")
        create_database(database)
        os.removedirs(database)

        os.rename( database + "_tmp",database)


        ch_files=glob.glob(file+"/*")

        for f in ch_files:
            print(f)
            tmp_filenmme = os.path.basename(f)
            os.rename(tmp_filenmme, "myfile.txt")




#             执行创建表
#              执行 清理表空间

