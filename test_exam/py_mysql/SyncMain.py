# -*- encoding:utf8 -*-_
from py_mysql.PyMysql import PyMysql


class SyncMain(object):
    def __init__(self,host, user, passwd, defaultdb):
        pymysql = PyMysql()

        self.mysql_conn= pymysql.newConnection("localhost", "root", "root", "line4_240201")
        pass

    def query(self,sql):
        return  self.mysql_conn.execute(sql)

