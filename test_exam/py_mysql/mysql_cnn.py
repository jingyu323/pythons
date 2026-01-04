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


def create_table():
    db = MySQLdb.connect("localhost", "root", "root", "line4_240201")
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # 如果数据表已经存在使用 execute() 方法删除表。
    cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")

    # 创建数据表SQL语句
    sql = """CREATE TABLE EMPLOYEE \
             ( \
                 FIRST_NAME CHAR(20) NOT NULL, \
                 LAST_NAME  CHAR(20), \
                 AGE        INT, \
                 SEX        CHAR(1), \
                 INCOME     FLOAT \
             )"""

    cursor.execute(sql)

    # 关闭数据库连接
    db.close()

def insert_data():
    db = MySQLdb.connect("localhost", "root", "root", "line4_240201")
    cursor = db.cursor()
    # SQL 插入语句
    sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
                                  LAST_NAME, AGE, SEX, INCOME)
             VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
    except:
        # Rollback in case there is any error
        db.rollback()

    # 关闭数据库连接
    db.close()


def select_data():
    db = MySQLdb.connect("localhost", "root", "root", "line4_240201")
    cursor = db.cursor()
    sql = """SELECT * FROM EMPLOYEE"""
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            print(row)
    except:
        print(  "Error: unable to fetch data")
    # 关闭数据库连接
    db.close()


def update_data():
    db = MySQLdb.connect("localhost", "root", "root", "line4_240201")
    cursor = db.cursor()
    # SQL 更新语句
    sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()

    # 关闭数据库连接
    db.close()

def delete_data():
    db = MySQLdb.connect("localhost", "root", "root", "line4_240201")
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 删除语句
    sql = "DELETE FROM EMPLOYEE WHERE AGE > %s" % (20)
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 提交修改
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()

    # 关闭连接
    db.close()





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