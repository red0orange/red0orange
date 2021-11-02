#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
"""
@author: red0orange
@file: database.py
@time:  7:35 PM
@desc:
"""
import datetime
import pymysql
from red0orange.utils import *


class DatabaseOperator(object):
    """
    封装对数据库的基本操作
    """
    def __init__(self, host_ip, user_name, user_pass, db_name):
        try:
            self._db = pymysql.connect(host_ip, user_name, user_pass, db_name)
        except:
            raise BaseException("连接失败，检查参数")

        self.cursor = self.db.cursor()
        pass

    @property
    def db(self):
        self._db.ping(reconnect=True)
        return self._db

    @classmethod
    def create_by_default(cls):
        """
        默认参数的db，这个是完全由代码指定
        Returns:

        """
        return cls("47.97.157.150", "red0orange", "asdfg123", "blood")

    @classmethod
    def create_by_normal(cls, host_ip, user_name, user_pass, db_name):
        """
        普通构建对象的方法
        Args:
            host_ip:
            user_name:
            user_pass:
            db_name:

        Returns:

        """
        return cls(host_ip, user_name, user_pass, db_name)

    @staticmethod
    def _get_column_names_types(cursor, table_name):
        sql = 'select column_name,data_type from information_schema.columns where table_name = "{}"'.format(table_name)
        cursor.execute(sql)
        names_types = cursor.fetchall()
        return [i[0] for i in names_types], [i[1] for i in names_types]

    @staticmethod
    def _get_column_names(cursor, table_name):
        return DatabaseOperator._get_column_names_types(cursor, table_name)[0]

    @staticmethod
    def _get_column_types(cursor, table_name):
        return DatabaseOperator._get_column_names_types(cursor, table_name)[1]

    def get_column_names(self, table_name):
        return self._get_column_types(self.cursor, table_name)

    def get_column_types(self, table_name):
        return self._get_column_types(self.cursor, table_name)

    @staticmethod
    def _get_column_type(cursor, table_name, column_name):
        """
        获取某个table的某一列的数据类型
        Args:
            cursor:
            table_name:
            column_name:

        Returns:

        """
        sql = f'select data_type from information_schema.columns where table_name="{table_name}" and column_name="{column_name}"'
        cursor.execute(sql)
        return cursor.fetchall()[0][0]

    @staticmethod
    def _modify_column_position(cursor, table_name, column_name, front_name=None):
        """
        改变数据表内某一列的位置
        Args:
            cursor:
            table_name:
            column_name:
            front_name:

        Returns:

        """
        column_type = DatabaseOperator._get_column_type(cursor, table_name, column_name)
        # TODO 这个问题要解决，现在是手动给varchar加东西
        # if column_type == 'varchar':
        #     column_type += '(10)'
        if front_name is None:
            sql = f'alter table {table_name} modify {column_name} {column_type} first'
            cursor.execute(sql)
        else:
            sql = f'alter table {table_name} modify {column_name} {column_type} after {front_name}'
            cursor.execute(sql)
        pass

    @staticmethod
    def _push_column_back(cursor, table_name, column_name, column_type):
        """
        在一个表后面加新字段
        Args:
            cursor:
            table_name:
            column_name:
            column_type:

        Returns:

        """
        sql = 'alter table {} add column {} {}'.format(table_name, column_name, column_type)
        cursor.execute(sql)
        pass

    @staticmethod
    def _modify_column_order(cursor, table_name, column_list):
        """
        按照column_list给table的column更换位置
        Args:
            cursor:
            table_name:
            column_list:

        Returns:

        """
        assert len(column_list) > 0, 'column list can not be empty'
        cur_list = DatabaseOperator._get_column_names(cursor, table_name)
        diff = get_list_subtraction(cur_list, column_list)
        for column_name in diff:
            # TODO 目前默认全部为varchar也就是不定长字符串类型
            DatabaseOperator._push_column_back(cursor, table_name, column_name, "text")

        last_name = None
        for i, column_name in enumerate(column_list):
            DatabaseOperator._modify_column_position(cursor, table_name, column_name, last_name)
            last_name = column_name

        pass

    @staticmethod
    def _insert_raw(cursor, table_name, data_dict):
        keys = list(data_dict.keys())
        values = list(data_dict.values())
        data_field_str = ','.join(keys)
        data_value_str = ','.join(['"' + str(i) + '"' for i in values])
        sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name, data_field_str, data_value_str)
        cursor.execute(sql)
        pass

    def modify_column_order(self, table_name, column_list):
        self._modify_column_order(self.cursor, table_name, column_list)
        self.db.commit()
        pass

    def insert_raw(self, table_name, data_dict):
        self._insert_raw(self.cursor, table_name, data_dict)
        self.db.commit()
        pass


class ExpResult(object):
    def __init__(self):
        self._init = False

        self.column_list = None
        self.db = None
        self.cur_dict = None
        self.dt = None
        self.table_name = None
        pass

    def init(self, table_name, column_list, description, id):
        self._init = True
        self.table_name = table_name

        self.db = DatabaseOperator.create_by_default()
        self.column_list = column_list
        self.column_list.insert(0, 'datetime')
        self.column_list.insert(1, 'description')
        self.column_list.insert(2, 'id')
        self.db.modify_column_order(table_name, self.column_list)

        self.cur_dict = {k: None for k in column_list}

        self.dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db.insert_raw(table_name, {'datetime': self.dt, 'description': description, 'id': id})
        pass

    def update(self, update_dict):
        if self._init:
            keys = list(update_dict.keys())
            assert if_list_contain_list(keys, self.column_list), "传入的更新dict，必须属于初始化时的列表"
            self.cur_dict.update(update_dict)

            for key in keys:
                # TODO 目前直接固定按照datetime索引进行更新，应该有更好的方式
                sql = 'update {} set {}="{}" where datetime="{}"'.format(self.table_name, key, update_dict[key], self.dt)
                self.db.cursor.execute(sql)
            self.db.db.commit()

            pass
        else:
            raise BaseException('not init')


exp_result = ExpResult()
