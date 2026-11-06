from datetime import datetime

from sqlalchemy import DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class RateLimitWindow(Base):
    """Search requests by one client in one fixed window; see app.core.rate_limit.

    UNLOGGED: every search rewrites a row here, and a crash may lose the counters,
    which at worst hands each client a fresh budget. Nothing else depends on them,
    so they are not worth the WAL. tests/test_schema_drift.py cannot see this
    prefix (Alembic's comparison ignores it), so tests/test_rate_limit_schema.py
    pins it against the catalog.

    Keyed by client_name rather than by key, so that the overlapping keys of a
    rotation share one budget. Its only index is the primary key: the table holds
    little more than one row per active client, so the sweep for expired windows
    needs none.
    """

    __tablename__ = "rate_limit_windows"
    __table_args__ = ({"prefixes": ["UNLOGGED"]},)

    client_name: Mapped[str] = mapped_column(Text, primary_key=True)
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    count: Mapped[int] = mapped_column(Integer, nullable=False)
