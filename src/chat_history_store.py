import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class ChatHistoryRecord(Base):
    __tablename__ = "chat_history"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    thread_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    route: Mapped[str] = mapped_column(String(32), index=True)
    query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class ChatHistoryStore:
    def __init__(self, database_url: str):
        if not database_url or not database_url.strip():
            raise ValueError("database_url cannot be empty")
        self.database_url = database_url.strip()
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(bind=self.engine)

    def add_entry(
        self,
        *,
        route: str,
        query: str,
        answer: str,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        record = ChatHistoryRecord(
            thread_id=thread_id,
            route=route,
            query=query,
            answer=answer,
            metadata_json=dict(metadata or {}),
        )
        with self.SessionLocal() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record.id

    def list_entries(
        self, *, thread_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(500, int(limit)))
        with self.SessionLocal() as session:
            stmt = select(ChatHistoryRecord).order_by(ChatHistoryRecord.created_at.desc())
            if thread_id:
                stmt = stmt.where(ChatHistoryRecord.thread_id == thread_id)
            stmt = stmt.limit(safe_limit)
            rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": row.id,
                "thread_id": row.thread_id,
                "route": row.route,
                "query": row.query,
                "answer": row.answer,
                "metadata": row.metadata_json,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    def delete_thread(self, thread_id: str) -> int:
        if not thread_id or not thread_id.strip():
            return 0
        with self.SessionLocal() as session:
            rows = session.query(ChatHistoryRecord).filter(
                ChatHistoryRecord.thread_id == thread_id.strip()
            )
            count = rows.count()
            rows.delete(synchronize_session=False)
            session.commit()
        return count
