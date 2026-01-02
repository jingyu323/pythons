import MySQLdb

# py_db_test
def connect_mysql_use_result():
    db = MySQLdb.connect("localhost", "root", "root", "line4_240201")

    db.query("""SELECT * FROM htgw_sync_main
             WHERE sync_id < 5""")


    r=db.use_result()

    row = r.fetchone()
    print(row)




    #
    # r.fetch_row()
    # print(r)
    # 逐行获取数据
    # while True:
    #     row = r.fetch_row()
    #     # row = r.fetch_row()
    #     if row is None:
    #         break
    #     print(row)






def db_version():

    # 打开数据库连接
    db = MySQLdb.connect("localhost", "root", "root", "line4_240201")

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # 使用execute方法执行SQL语句
    cursor.execute("SELECT VERSION()")

    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchone()

    print("Database version : %s " % data)


    # 关闭数据库连接
    db.close()


def connect_mysql_cursor_query():
    db = MySQLdb.connect("localhost", "root", "root", "line4_240201")



    try:
        # 创建游标对象
        cursor = db.cursor()

        # 执行查询
        cursor.execute("SELECT * FROM htgw_sync_main WHERE sync_id < 5")

        # 获取所有记录
        results = cursor.fetchall()

        # 打印记录
        for row in results:
            print(row)

            fname = row[0]
            lname = row[1]
            age = row[2]
            sex = row[3]
            income = row[4]
            # 打印结果
            print("fname=%s,lname=%s,age=%s,sex=%s,income=%s" %  (fname, lname, age, sex, income))


    finally:
        # 关闭游标和连接
        cursor.close()
        db.close()


if __name__ == '__main__':
    connect_mysql_use_result()