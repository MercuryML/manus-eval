import pandas as pd
import polars as pl
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from urllib.parse import urlparse, parse_qs
import datetime
import sys
import os
from typing import Optional, List, Dict, Union


def eprint(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}", file=sys.stderr)


class PostgreSQLData:
    """
    PostgreSQL data operation utility class
    
    Usage example:
    from pg_data import PostgreSQLData
    pg = PostgreSQLData.load_from_env()
    # Read data as polars DataFrame
    polars_df = pg.read_sql_to_pldf("SELECT * FROM table_name LIMIT 10")
    # Read data as pandas DataFrame
    pandas_df = pg.read_sql_to_df("SELECT * FROM table_name LIMIT 10")
    # Write data
    pg.write_df(df, "table_name", write_mode="overwrite")
    """
    
    def __init__(self, uri: str, meta: dict) -> None:
        self.uri = uri
        self.meta = meta
        self.config = PostgreSQLData.parse_postgresql_url(uri,meta)
        self._total_count: int = ...
        self._engine = None
        self._connection = None

    @staticmethod
    def load_from_env() -> "PostgreSQLData":
        """Load PostgreSQL connection configuration from environment variables"""
        meta = {
            "postgresUser": os.getenv("postgres_user") or os.getenv("POSTGRES_USER"),
            "postgresPassword": os.getenv("postgres_password") or os.getenv("POSTGRES_PASSWORD"),
            "postgresHost": os.getenv("postgres_host") or os.getenv("POSTGRES_HOST", "localhost"),
            "postgresPort": os.getenv("postgres_port") or os.getenv("POSTGRES_PORT", "5432"),
            "postgresDatabase": os.getenv("postgres_database") or os.getenv("POSTGRES_DB"),
            "postgresSslmode": os.getenv("postgres_sslmode") or os.getenv("POSTGRES_SSLMODE"),
        }
        uri = os.getenv("postgres_uri") or os.getenv("POSTGRES_URI")
        if not uri:
            # If no URI, try to construct from separate environment variables
            if meta["postgresUser"] and meta["postgresPassword"] and meta["postgresDatabase"]:
                uri = f"postgresql://{meta['postgresUser']}:{meta['postgresPassword']}@{meta['postgresHost']}:{meta['postgresPort']}/{meta['postgresDatabase']}"
                if meta['postgresSslmode']:
                    uri += f"?sslmode={meta['postgresSslmode']}"
            else:
                eprint("Environment variable error: postgres_uri not set, or missing required connection parameters")
                raise ValueError("postgres_uri not set or connection parameters incomplete")
        
        if not all([meta.get("postgresUser"), meta.get("postgresPassword")]):
            eprint("Environment variable error: postgres_user or postgres_password not set")
            raise ValueError("postgres_user or postgres_password not set")
            
        return PostgreSQLData(uri, meta=meta)
    
    @staticmethod
    def parse_postgresql_url(url:str,meta:dict) -> dict:
        """Parse PostgreSQL connection URL"""
        parsed_url = urlparse(url)
        
        # Extract basic information
        scheme = parsed_url.scheme
        
        # Handle authentication information
        if parsed_url.username and parsed_url.password:
            user = parsed_url.username
            password = parsed_url.password
        else:
            user = meta.get("postgresUser")
            password = meta.get("postgresPassword")
            
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 5432
        database = parsed_url.path.lstrip("/") if parsed_url.path else None
        
        # Parse query parameters
        params = parse_qs(parsed_url.query)
        params = {k: v[0] for k, v in params.items()}
        
        config = {
            "scheme": scheme,
            "user": user,
            "password": password,
            "host": host,
            "port": port,
            "database": database,
            "sslmode": params.get("sslmode", "prefer"),
            **params,
        }
        return config

    def get_connection(self):
        """Get database connection"""
        if self._connection is None or self._connection.closed:
            config = self.config
            self._connection = psycopg2.connect(
                host=config["host"],
                port=config["port"],
                user=config["user"],
                password=config["password"],
                database=config["database"],
                sslmode=config.get("sslmode", "prefer"),
                cursor_factory=RealDictCursor
            )
        return self._connection

    def get_engine(self):
        """Get SQLAlchemy engine"""
        if self._engine is None:
            # Build connection string
            config = self.config
            connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
            if config.get("sslmode"):
                connection_string += f"?sslmode={config['sslmode']}"
            
            self._engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        return self._engine

    def execute_query(self, sql: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute query and return results"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                if cursor.description:
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return []
        except Exception as e:
            conn.rollback()
            raise e
    
    def read_sql_to_dict(self, sql: str, params: Optional[Dict] = None) -> Dict[str, List]:
        """Execute SQL query and return dictionary format results"""
        eprint(f"== Loading PostgreSQL dataset {self.uri} SQL: {sql} ==")
        
        engine = self.get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            rows = result.fetchall()
            
            if not rows:
                return {}
                
            # Get column names
            column_names = list(result.keys())
            
            # Convert to dictionary format
            result_dict = {col: [] for col in column_names}
            for row in rows:
                for i, col in enumerate(column_names):
                    result_dict[col].append(row[i])
                    
        return result_dict

    def read_sql_to_df(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute SQL query and return pandas DataFrame"""
        eprint(f"== Loading PostgreSQL dataset {self.uri} SQL: {sql} ==")
        
        engine = self.get_engine()
        df = pd.read_sql_query(text(sql), engine, params=params or {})
        return df

    def read_sql_to_pldf(self, sql: str, params: Optional[Dict] = None) -> pl.DataFrame:
        """Execute SQL query and return polars DataFrame"""
        eprint(f"== Loading PostgreSQL dataset {self.uri} SQL: {sql} ==")
        
        pandas_df = self.read_sql_to_df(sql, params)
        polars_df = pl.from_pandas(pandas_df)
        return polars_df

    @staticmethod
    def _pandas_to_postgresql_type(df: pd.DataFrame, col_name: str, dtype) -> str:
        """Convert pandas data type to PostgreSQL data type"""
        # Check if column contains NULL values
        has_null = df[col_name].isna().any() # noqa
        
        # Integer types
        if pd.api.types.is_integer_dtype(dtype):
            if dtype == np.int16:
                return "SMALLINT"
            elif dtype == np.int32:
                return "INTEGER"
            elif dtype == np.int64:
                return "BIGINT"
            else:
                return "INTEGER"
        
        # Float types
        elif pd.api.types.is_float_dtype(dtype):
            if dtype == np.float32:
                return "REAL"
            else:
                return "DOUBLE PRECISION"
        
        # Boolean type
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        
        # Datetime type
        elif pd.api.types.is_datetime64_dtype(dtype):
            return "TIMESTAMP"
        
        # Date type
        elif isinstance(dtype, pd.PeriodDtype) or (
            pd.api.types.is_object_dtype(dtype) and 
            all(isinstance(x, datetime.date) for x in df[col_name].dropna())
        ):
            return "DATE"
        
        # Categorical type
        elif isinstance(dtype, pd.CategoricalDtype):
            return "TEXT"
        
        # More precise numeric type judgment
        elif pd.api.types.is_numeric_dtype(dtype):
            return "NUMERIC"
        
        # Default to text type
        else:
            # Check string length to decide between VARCHAR and TEXT
            if pd.api.types.is_object_dtype(dtype):
                max_length = df[col_name].astype(str).str.len().max()
                if pd.isna(max_length) or max_length <= 255:
                    return "VARCHAR(255)"
                else:
                    return "TEXT"
            return "TEXT"

    def _save_dataframe_to_postgresql(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        table_name: str,
        create_table: bool = True,
        if_exists: str = "replace",  # 'replace', 'append', 'fail'
        batch_size: int = 10000,
        index: bool = False,
        method: str = "multi",
        tag: Optional[str] = None,
        primary_key: Optional[List[str]] = None,
        indexes: Optional[List[str]] = None,
    ):
        """
        Save DataFrame to PostgreSQL database
        
        Parameters:
            df: pandas or polars DataFrame
            table_name: table name
            create_table: whether to create table
            if_exists: behavior when table exists ('replace', 'append', 'fail')
            batch_size: batch insert size
            index: whether to include index
            method: insert method ('multi', None)
            tag: log tag
            primary_key: primary key column list
            indexes: list of columns to create indexes for
        """
        # Convert to pandas DataFrame
        if isinstance(df, pl.DataFrame):
            pandas_df = df.to_pandas()
        else:
            pandas_df = df
            
        tag = tag or if_exists
        total_rows = len(pandas_df)
        
        eprint(f"Starting to save data [{tag}]: {pandas_df.shape} to {self.uri}/{table_name} mode: {if_exists}")
        
        engine = self.get_engine()
        
        # If replace mode and need to create table, drop table first
        if if_exists == "replace" and create_table:
            from psycopg2 import sql # noqa
            # For SQLAlchemy, we need to manually construct safe SQL
            drop_sql = f"DROP TABLE IF EXISTS {table_name}"  # Using f-string is safe here because SQLAlchemy handles it
            with engine.connect() as conn:
                conn.execute(text(drop_sql))
                conn.commit()
        
        # Use pandas to_sql method
        pandas_df.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=index,
            method=method,
            chunksize=batch_size,
        )
        
        # Create primary key
        if primary_key and create_table:
            with engine.connect() as conn:
                # For table names and column names, we use quotes here to ensure safety
                quoted_table = f'"{table_name}"'
                quoted_columns = ', '.join([f'"{col}"' for col in primary_key])
                pk_sql = f"ALTER TABLE {quoted_table} ADD PRIMARY KEY ({quoted_columns})"
                try:
                    conn.execute(text(pk_sql))
                    conn.commit()
                    eprint(f"Primary key created: {primary_key}")
                except Exception as e:
                    eprint(f"Failed to create primary key: {e}", "WARNING")
        
        # Create indexes
        if indexes and create_table:
            with engine.connect() as conn:
                for col in indexes:
                    quoted_table = f'"{table_name}"'
                    quoted_col = f'"{col}"'
                    index_name = f"idx_{table_name}_{col}"
                    index_sql = f"CREATE INDEX {index_name} ON {quoted_table} ({quoted_col})"
                    try:
                        conn.execute(text(index_sql))
                        conn.commit()
                        eprint(f"Index created: {col}")
                    except Exception as e:
                        eprint(f"Failed to create index {col}: {e}", "WARNING")
        
        eprint(f"Successfully saved DataFrame[{tag}/{total_rows}] {if_exists} to PostgreSQL table '{table_name}'")

    def write_df(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        table_name: str,
        *,
        write_mode: str = "overwrite",
        primary_key: Optional[List[str]] = None,
        indexes: Optional[List[str]] = None,
        tag: Optional[str] = None,
    ):
        """
        Write DataFrame to PostgreSQL table
        
        Parameters:
            df: DataFrame to write
            table_name: target table name
            write_mode: write mode ('overwrite', 'append')
            primary_key: primary key column list
            indexes: list of columns to create indexes for
            tag: log tag
        """
        if_exists = "replace" if write_mode == "overwrite" else "append"
        
        self._save_dataframe_to_postgresql(
            df,
            table_name,
            create_table=True,
            if_exists=if_exists,
            tag=tag,
            primary_key=primary_key,
            indexes=indexes,
        )

    def list_tables(self, schema: str = "public") -> List[str]:
        """List tables in the database"""
        sql = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        
        conn = self.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, (schema,))
            tables = [row["table_name"] for row in cursor.fetchall()]
            
        return tables

    def count_rows(self, table_name: Optional[str] = None) -> int:
        """Get the number of rows in the table"""
        if table_name is None:
            if not hasattr(self, 'default_table'):
                raise ValueError("Table name not specified and no default table exists")
            table_name = self.default_table
            
        from psycopg2 import sql
        count_sql = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))
        conn = self.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(count_sql)
            result = cursor.fetchone()
            return result['count'] if result else 0

    def create_index(self, table_name: str, column_name: str, index_type: str = "btree"):
        """Create index"""
        from psycopg2 import sql
        
        index_name = f"idx_{table_name}_{column_name}"
        create_index_sql = sql.SQL("CREATE INDEX {} ON {} USING {} ({})").format(
            sql.Identifier(index_name),
            sql.Identifier(table_name),
            sql.SQL(index_type),
            sql.Identifier(column_name)
        )
        
        conn = self.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(create_index_sql)
            conn.commit()
            eprint(f"Index created: {index_name}")

    def drop_index(self, index_name: str):
        """Delete index"""
        from psycopg2 import sql
        
        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(index_name))
        
        conn = self.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(drop_index_sql)
            conn.commit()
            eprint(f"Index deleted: {index_name}")

    @staticmethod
    def parser_date_to_str(df: pl.DataFrame) -> pl.DataFrame:
        """Convert date columns to string format"""
        date_cols = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if dtype in (pl.Date, pl.Datetime)
        ]
        
        if date_cols:
            df = df.with_columns([
                pl.col(c).dt.strftime("%Y-%m-%d").alias(c) for c in date_cols
            ])
        
        return df

    def close(self):
        """Close connection"""
        if self._connection and not self._connection.closed:
            self._connection.close()
        if self._engine:
            self._engine.dispose()