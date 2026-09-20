import psycopg2
from pgvector.psycopg2 import register_vector

from app.config import settings


def get_db_connection():
    conn = psycopg2.connect(settings.database_url)
    register_vector(conn)
    return conn