import aiosqlite


class Drovix(object):

    def __init__(self, db: str):
        self.db = db

    async def __aenter__(self):
        self.conn = await aiosqlite.connect(self.db)
        self.cursor = await self.conn.cursor()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cursor.close()
        await self.conn.close()

    async def create(self, table_name: str, *args):
        columns = ", ".join(args)
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})"
        await self.cursor.execute(sql)
        await self.conn.commit()

    async def insert(self, table_name: str, column_names: list, values: tuple):
        placeholders = ", ".join(["?"] * len(values))
        columns = ", ".join(column_names)
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        await self.cursor.execute(sql, values)
        await self.conn.commit()

    async def demand(self, table_name: str) -> tuple:
        async with self.conn.execute(f"SELECT * FROM {table_name}") as cursor:
            return await cursor.fetchall()


if __name__ == '__main__':
    pass
