import os

import MySQLdb
import pymysql

DBHOST = "localhost"
DBUSER = "root"
DBPASS = "root"
DBNAME = "hih"

def MySQLdb_conn():
    connection = MySQLdb.connect(
        host=DBHOST,
        user=DBUSER,
        passwd=DBPASS,
        db=DBNAME,
    )

    cursor = connection.cursor()
    cursor.execute("select @@version")
    version = cursor.fetchone()
    if version:
        print('Running version: ', version)
    else:
        print('Not connected.')


def  pymysql_conn():

    conn = pymysql.connect(
        user=DBUSER,
        password=DBPASS,
        # MySQL的默认端口为3306
        port=3306,
        # 本机地址为127.0.0.1或localhost
        host=DBHOST,
        # 指定使用的数据库
        init_command='use  '+DBNAME
    )
    # 创建游标对象
    cur = conn.cursor()

    cur.execute("select @@version ")
    version = cur.fetchone()
    if version:
        print('Running version: ', version)
    else:
        print('Not connected.')
    conn.close()


if "__main__" == __name__ :
    pymysql_conn()