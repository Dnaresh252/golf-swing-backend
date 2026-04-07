import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# Load .env so DATABASE_URL is available before the engine is created
# ---------------------------------------------------------------------------
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Inject DATABASE_URL from environment into the ini config.
# This satisfies the %(DATABASE_URL)s interpolation in alembic.ini.
database_url = os.environ.get("DATABASE_URL", "")

# Alembic needs a sync driver; swap asyncpg for psycopg2 if present.
if "+asyncpg" in database_url:
    database_url = database_url.replace("+asyncpg", "")

config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

# ---------------------------------------------------------------------------
# Logging — honour alembic.ini [loggers] section if present
# ---------------------------------------------------------------------------
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import Base + every model so Alembic can detect all tables
# ---------------------------------------------------------------------------
from app.database import Base  # noqa: E402
from app.models import (  # noqa: E402, F401
    Avatar,
    Coach,
    CoachNotes,
    CorrectedVideo,
    DiscountCode,
    ResultsVideo,
    SocialSharing,
    Submission,
    SubmissionFile,
    User,
)

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Helper: build URL with sync driver for Alembic
# ---------------------------------------------------------------------------

def _get_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    return url.replace("+asyncpg", "") if "+asyncpg" in url else url


# ---------------------------------------------------------------------------
# Offline migrations (generate SQL without connecting)
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine — no DB
    connection is opened.  Useful for generating SQL scripts.
    """
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations (apply changes to a live database)
# ---------------------------------------------------------------------------

def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an Engine and associates a connection with the migration context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
