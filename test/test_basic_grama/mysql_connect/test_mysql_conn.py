import os
import traceback

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


def create_table():
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
    cur = conn.cursor()
    SQL = 'create table if not exists info(' \
          'id int primary key,' \
          'title char not null,' \
          'photo_src char not null)'

    try:
        # 开启一个事务
        conn.begin()
        # 设置将执行的SQL语句
        cur.execute(SQL)
        # 提交事务
        conn.commit()
    except Exception:
        print('【初始化失败（表）】')
        # 打印错误信息
        print('    ', traceback.print_exc())





if "__main__" == __name__ :
    create_table()