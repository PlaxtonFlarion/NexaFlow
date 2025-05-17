#    ____      _     _      _
#   / ___|   _| |__ (_) ___| | ___
#  | |  | | | | '_ \| |/ __| |/ _ \
#  | |__| |_| | |_) | | (__| |  __/
#   \____\__,_|_.__/|_|\___|_|\___|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import typing
import aiosqlite
from pathlib import Path
from nexaflow import const


class DB(object):
    """
    异步 SQLite 数据库操作类。

    该类封装了 SQLite 的异步连接、表创建、数据插入与查询等基本功能，主要用于在分析任务中进行数据持久化存储。

    Parameters
    ----------
    db : str
        SQLite 数据库文件的路径。

    Notes
    -----
    - 使用 `async with` 管理数据库连接和资源释放。
    - 所有操作默认在 `const.DB_TABLE_NAME` 表中进行。
    """

    def __init__(self, db: typing.Union[str, "Path"]):
        """
        初始化类实例，并设置数据库路径。

        Parameters
        ----------
        db : Union[str, Path]
            数据库文件的路径，可以是字符串形式的路径，或 `pathlib.Path` 对象。
            此路径通常用于后续数据库连接或读写操作。
        """
        self.db = db

    async def __aenter__(self) -> "DB":
        """
        异步上下文管理器入口，建立数据库连接。

        Returns
        -------
        DB
            返回当前数据库操作对象。
        """
        self.conn = await aiosqlite.connect(self.db)
        self.cursor = await self.conn.cursor()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        异步上下文管理器退出，关闭数据库连接。

        Parameters
        ----------
        exc_type : Type[BaseException]
            异常类型。

        exc_val : BaseException
            异常实例。

        exc_tb : TracebackType
            异常堆栈信息。
        """
        await self.cursor.close()
        await self.conn.close()

    async def create(self, column_list: list) -> None:
        """
        创建表结构（如果尚不存在）。

        Parameters
        ----------
        column_list : list
            表字段列表（不含主键 id）。

        Notes
        -----
        - 会自动添加自增主键字段 `id`。
        - 若表已存在，则跳过创建。

        Workflow
        --------
        1. 拼接字段语句。
        2. 构造 CREATE TABLE SQL 语句。
        3. 执行并提交事务。
        """
        columns = ", ".join(column_list)
        sql = f"CREATE TABLE IF NOT EXISTS {const.DB_TABLE_NAME} (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})"
        await self.cursor.execute(sql)
        await self.conn.commit()

    async def insert(self, column_list: list, values: list) -> None:
        """
        插入数据到表中。

        Parameters
        ----------
        column_list : list
            列名列表，顺序应与 values 一致。
        values : list
            待插入的数据列表。

        Raises
        ------
        ValueError
            如果 `column_list` 和 `values` 长度不一致。

        Workflow
        --------
        1. 拼接 INSERT INTO SQL 语句。
        2. 使用参数化方式插入数据。
        3. 提交事务。
        """
        placeholders = ", ".join(["?"] * len(values))
        columns = ", ".join(column_list)
        sql = f"INSERT INTO {const.DB_TABLE_NAME} ({columns}) VALUES ({placeholders})"
        await self.cursor.execute(sql, values)
        await self.conn.commit()

    async def demand(self) -> tuple:
        """
        查询当前表中的所有记录。

        Returns
        -------
        tuple
            包含数据库所有行的元组列表。

        Workflow
        --------
        1. 执行 SELECT * 查询。
        2. 异步提取所有结果。
        """
        async with self.conn.execute(f"SELECT nest FROM {const.DB_TABLE_NAME}") as cursor:
            return await cursor.fetchall()


if __name__ == '__main__':
    pass
