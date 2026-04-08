"""SQLAlchemy engine, session factory, and database initialisation."""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DB_CONNECT_ARGS, DB_ECHO, DB_PATH

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args=DB_CONNECT_ARGS,
    echo=DB_ECHO,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


def init_db() -> None:
    """Create all tables if they don't exist."""
    # Import models so Base.metadata knows about them
    from . import models as _models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
