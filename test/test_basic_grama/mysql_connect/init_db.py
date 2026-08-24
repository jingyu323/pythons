import traceback
from subprocess import Popen, PIPE

import mysql.connector
DBHOST = "localhost"
DBUSER = "root"
DBPASS = "root"
DBNAME = "hih"
port = "3306"
def init_db(sql_file_path,db_name):
    process = Popen(' mysql.exe -h%s -P%s -u%s -p%s %s' % (DBHOST, port, DBUSER, DBPASS, db_name),
                    stdout=PIPE, stdin=PIPE, shell=True)
    output = process.communicate('source ' + sql_file_path)



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



if __name__ == '__main__':

    dbname="test333211"

    create_database(dbname)

    init_db("E:\study\git\pythons\test\test_basic_grama\init_database\all.sql",dbname)