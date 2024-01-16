import sqlite3


class DataBase(object):

    def __init__(self, db: str):
        self.conn = sqlite3.connect(db)
        self.cursor = self.conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
        self.conn.close()

    def create(self, table_name: str, *args):
        columns = ', '.join(args)
        sql = f'CREATE TABLE IF NOT EXISTS {table_name} ({columns})'
        self.cursor.execute(sql)
        self.conn.commit()

    def insert(self, table_name: str, *args):
        placeholders = ', '.join(['?'] * len(args))
        sql = f'INSERT INTO {table_name} VALUES ({placeholders})'
        self.cursor.execute(sql, args)
        self.conn.commit()

    def demand(self, table_name: str) -> tuple:
        data = self.cursor.execute(f'SELECT * FROM {table_name}')
        return data.fetchall()[0]


if __name__ == '__main__':
    pass
