"""SQLAlchemy ORM models for PostgreSQL – document and chunk metadata."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Document(Base):
    """Uploaded PDF document metadata."""

    __tablename__ = "documents"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: str = Column(String(512), nullable=False)
    upload_path: str = Column(Text, nullable=False)
    page_count: int = Column(Integer, nullable=False, default=0)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("SpatialChunk", back_populates="document", cascade="all, delete-orphan")


class SpatialChunk(Base):
    """A text chunk extracted from a PDF with spatial (x, y) metadata."""

    __tablename__ = "spatial_chunks"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_no: int = Column(Integer, nullable=False)
    text: str = Column(Text, nullable=False)

    # Bounding-box coordinates (PDF units)
    x0: float = Column(Float, nullable=False)
    y0: float = Column(Float, nullable=False)
    x1: float = Column(Float, nullable=False)
    y1: float = Column(Float, nullable=False)

    # Detected nearby handwritten marker (star, underline, circle, …)
    marker_type: str | None = Column(String(64), nullable=True)
    marker_distance: float | None = Column(Float, nullable=True)

    # Qdrant point id for cross-reference
    qdrant_point_id: str | None = Column(String(128), nullable=True)

    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
