#
#    ____      _     _      _
#   / ___|   _| |__ (_) ___| | ___
#  | |  | | | | '_ \| |/ __| |/ _ \
#  | |__| |_| | |_) | | (__| |  __/
#   \____\__,_|_.__/|_|\___|_|\___|
#

import aiosqlite
from nexaflow import const


class DB(object):

    def __init__(self, db: str):
        """
        数据库操作类，封装了异步数据库连接和基本地增删查改功能。

        用法:
            该类支持使用 `async with` 语句，确保在操作完成后自动关闭数据库连接。

        参数:
            db (str): 数据库文件的路径。
        """
        self.db = db

    async def __aenter__(self) -> "DB":
        """
        异步上下文管理器入口方法。

        功能:
            打开数据库连接并创建游标，用于执行 SQL 操作。

        返回:
            DB: 返回当前实例，允许在 `async with` 块中调用其他方法。

        异常:
            aiosqlite.Error: 如果连接数据库失败，可能抛出数据库相关异常。
        """
        self.conn = await aiosqlite.connect(self.db)
        self.cursor = await self.conn.cursor()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        异步上下文管理器退出方法。

        功能:
            关闭数据库游标和连接，确保资源释放。

        参数:
            exc_type: 异常类型。
            exc_val: 异常值。
            exc_tb: 异常追踪信息。

        异常:
            aiosqlite.Error: 如果关闭游标或连接失败，可能抛出数据库相关异常。
        """
        await self.cursor.close()
        await self.conn.close()

    async def create(self, column_list: list) -> None:
        """
        创建数据库表，如果表不存在。

        功能:
            根据提供的列名列表创建表结构。如果表已存在，则不会重复创建。

        参数:
            column_list (list): 包含要创建的表列名的列表，每个元素格式为 "<column_name> <data_type>"。

        异常:
            aiosqlite.Error: 如果创建表的 SQL 执行失败，可能抛出数据库相关异常。
        """
        columns = ", ".join(column_list)
        sql = f"CREATE TABLE IF NOT EXISTS {const.DB_TABLE_NAME} (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})"
        await self.cursor.execute(sql)
        await self.conn.commit()

    async def insert(self, column_list: list, values: list) -> None:
        """
        插入数据记录到数据库表中。

        功能:
            根据列名列表和对应的值，将一条记录插入到数据库表中。

        参数:
            column_list (list): 包含要插入的列名的列表。
            values (list): 包含要插入的值的列表，顺序应与列名对应。

        异常:
            aiosqlite.Error: 如果插入数据的 SQL 执行失败，可能抛出数据库相关异常。
        """
        placeholders = ", ".join(["?"] * len(values))
        columns = ", ".join(column_list)
        sql = f"INSERT INTO {const.DB_TABLE_NAME} ({columns}) VALUES ({placeholders})"
        await self.cursor.execute(sql, values)
        await self.conn.commit()

    async def demand(self) -> tuple:
        """
        查询数据库表中的所有记录。

        功能:
            执行查询操作，返回表中的所有记录。

        返回:
            tuple: 包含所有记录的元组，每条记录为一个元组。

        异常:
            aiosqlite.Error: 如果查询数据的 SQL 执行失败，可能抛出数据库相关异常。
        """
        async with self.conn.execute(f"SELECT * FROM {const.DB_TABLE_NAME}") as cursor:
            return await cursor.fetchall()


if __name__ == '__main__':
    pass
