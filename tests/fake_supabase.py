"""In-memory stand-in for the Supabase query builder used by harvest.trading_store.

Supports the subset of the postgrest API the store uses: select, insert,
upsert(on_conflict=...), update, delete, eq, neq, in_, is_, gte, lte,
not_.<filter>, order and limit. Rows are copied in and out like over HTTP.
"""
import copy
import itertools


# Primary keys from pipeline/premium_trading.sql; "id" columns are bigserial.
PRIMARY_KEYS = {
    'premium_users': ('email',),
    'donations': ('provider', 'provider_tx_id'),
    'trading_runs': ('id',),
    'trading_setups': ('tool', 'scope'),
    'trading_signals': ('tool', 'symbol', 'as_of'),
    'positions': ('id',),
    'user_settings': ('email',),
    'alert_log': ('email', 'kind', 'ref'),
}
SERIAL_TABLES = {'trading_runs', 'positions'}


class Result:
    def __init__(self, data):
        self.data = data


class DuplicateKeyError(Exception):
    pass


class Query:
    def __init__(self, db, table):
        self.db = db
        self.table = table
        self.action = 'select'
        self.payload = None
        self.on_conflict = None
        self.filters = []
        self.orders = []
        self.row_limit = None
        self._negate = False

    # Actions
    def select(self, columns='*'):
        self.action = 'select'
        return self

    def insert(self, rows):
        self.action, self.payload = 'insert', rows
        return self

    def upsert(self, rows, on_conflict=None):
        self.action, self.payload, self.on_conflict = 'upsert', rows, on_conflict
        return self

    def update(self, values):
        self.action, self.payload = 'update', values
        return self

    def delete(self):
        self.action = 'delete'
        return self

    # Filters
    @property
    def not_(self):
        self._negate = True
        return self

    def _filter(self, test):
        negate, self._negate = self._negate, False
        self.filters.append((lambda row: not test(row)) if negate else test)
        return self

    def eq(self, column, value):
        return self._filter(lambda row: row.get(column) == value)

    def neq(self, column, value):
        return self._filter(lambda row: row.get(column) != value)

    def in_(self, column, values):
        values = list(values)
        return self._filter(lambda row: row.get(column) in values)

    def is_(self, column, value):
        assert value in ('null', None), 'fake only supports is_(column, "null")'
        return self._filter(lambda row: row.get(column) is None)

    def gte(self, column, value):
        return self._filter(lambda row: row.get(column) is not None and row[column] >= value)

    def lte(self, column, value):
        return self._filter(lambda row: row.get(column) is not None and row[column] <= value)

    def order(self, column, desc=False):
        self.orders.append((column, desc))
        return self

    def limit(self, count):
        self.row_limit = count
        return self

    # Execution
    def _matches(self):
        return [row for row in self.db.rows(self.table) if all(test(row) for test in self.filters)]

    def execute(self):
        self.db.executed.append((self.table, self.action))
        handler = getattr(self, f'_execute_{self.action}')
        return Result(copy.deepcopy(handler()))

    def _execute_select(self):
        rows = self._matches()
        for column, desc in reversed(self.orders):
            present = [row for row in rows if row.get(column) is not None]
            missing = [row for row in rows if row.get(column) is None]
            rows = sorted(present, key=lambda row: row[column], reverse=desc) + missing
        return rows[: self.row_limit] if self.row_limit is not None else rows

    def _execute_insert(self):
        rows = self.payload if isinstance(self.payload, list) else [self.payload]
        stored = []
        for row in rows:
            row = self.db.with_serial(self.table, copy.deepcopy(row))
            if self.db.find(self.table, row) is not None:
                raise DuplicateKeyError(f'duplicate key in {self.table}: {row}')
            self.db.rows(self.table).append(row)
            stored.append(row)
        return stored

    def _execute_upsert(self):
        rows = self.payload if isinstance(self.payload, list) else [self.payload]
        keys = tuple(self.on_conflict.split(',')) if self.on_conflict else PRIMARY_KEYS[self.table]
        stored = []
        for row in rows:
            row = copy.deepcopy(row)
            existing = next(
                (old for old in self.db.rows(self.table) if all(old.get(k) == row.get(k) for k in keys)),
                None,
            )
            if existing is None:
                row = self.db.with_serial(self.table, row)
                self.db.rows(self.table).append(row)
                stored.append(row)
            else:
                existing.update(row)
                stored.append(existing)
        return stored

    def _execute_update(self):
        rows = self._matches()
        for row in rows:
            row.update(copy.deepcopy(self.payload))
        return rows

    def _execute_delete(self):
        doomed = self._matches()
        doomed_ids = {id(row) for row in doomed}
        kept = [row for row in self.db.rows(self.table) if id(row) not in doomed_ids]
        self.db.tables[self.table] = kept
        return doomed


class FakeSupabase:
    def __init__(self, tables=None):
        self.tables = {name: copy.deepcopy(rows) for name, rows in (tables or {}).items()}
        self.executed = []
        self._ids = itertools.count(1)

    def table(self, name):
        if name not in PRIMARY_KEYS:
            raise KeyError(f'unknown table {name}')
        return Query(self, name)

    def rows(self, name):
        return self.tables.setdefault(name, [])

    def with_serial(self, name, row):
        if name in SERIAL_TABLES and row.get('id') is None:
            row['id'] = next(self._ids)
        return row

    def find(self, name, row):
        keys = PRIMARY_KEYS[name]
        return next(
            (old for old in self.rows(name) if all(old.get(k) == row.get(k) for k in keys)),
            None,
        )
