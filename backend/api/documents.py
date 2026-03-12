"""Document upload and management endpoints."""

import logging
import os
import shutil
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import Document, SpatialChunk
from models.schemas import DocumentOut, SpatialChunkPayload
from services.db import get_db
from services.spatial_chunker import process_pdf
from services.vector_store import delete_by_document, upsert_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
) -> DocumentOut:
    """Upload a PDF, extract spatial chunks, and index in Qdrant."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Persist file to disk
    upload_dir = os.path.abspath(settings.upload_dir)
    os.makedirs(upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}.pdf"
    file_path = os.path.join(upload_dir, safe_name)

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # Extract spatial chunks
        chunks, page_count = process_pdf(file_path)
        logger.info("Extracted %d chunks from '%s'", len(chunks), file.filename)

        # Save document record
        doc = Document(
            id=uuid.UUID(file_id),
            filename=file.filename,
            upload_path=file_path,
            page_count=page_count,
        )
        db.add(doc)

        # Build Qdrant payloads
        payloads = [
            SpatialChunkPayload(
                chunk_id=c.id,
                document_id=file_id,
                page_no=c.page_no,
                text=c.text,
                x0=c.x0,
                y0=c.y0,
                x1=c.x1,
                y1=c.y1,
                marker_type=c.marker_type,
                marker_distance=c.marker_distance,
            )
            for c in chunks
        ]

        # Upsert into Qdrant (batched internally)
        point_ids = await upsert_chunks(payloads)

        # Bulk-insert chunk records into PostgreSQL
        db.add_all(
            [
                SpatialChunk(
                    id=uuid.UUID(chunk_data.id),
                    document_id=uuid.UUID(file_id),
                    page_no=chunk_data.page_no,
                    text=chunk_data.text,
                    x0=chunk_data.x0,
                    y0=chunk_data.y0,
                    x1=chunk_data.x1,
                    y1=chunk_data.y1,
                    marker_type=chunk_data.marker_type,
                    marker_distance=chunk_data.marker_distance,
                    qdrant_point_id=pid,
                )
                for chunk_data, pid in zip(chunks, point_ids)
            ]
        )

        await db.commit()
        await db.refresh(doc)
        return DocumentOut.model_validate(doc)

    except Exception as exc:
        logger.error("Upload failed for '%s': %s", file.filename, exc, exc_info=True)
        # Clean up uploaded file so disk doesn't accumulate orphaned PDFs
        if os.path.exists(file_path):
            os.remove(file_path)
        # Best-effort: remove any Qdrant points already written
        try:
            await delete_by_document(file_id)
        except Exception:
            pass
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")


@router.get("/", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentOut]:
    """Return all uploaded documents."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return [DocumentOut.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentOut:
    """Return metadata for a single document."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentOut.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document and all its chunks from both stores."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove from Qdrant
    await delete_by_document(str(document_id))

    # Remove file from disk
    if os.path.exists(doc.upload_path):
        os.remove(doc.upload_path)

    # Cascade delete in PostgreSQL
    await db.delete(doc)
    await db.commit()


@router.get("/{document_id}/file")
async def serve_document_file(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Serve the raw PDF file for the frontend viewer."""
    from urllib.parse import quote
    from fastapi.responses import FileResponse

    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not os.path.exists(doc.upload_path):
        raise HTTPException(status_code=404, detail="File not found on disk.")

    # RFC 5987 encoding for non-ASCII (e.g. Korean) filenames in HTTP headers
    encoded_name = quote(doc.filename, safe="")
    content_disposition = f"inline; filename*=UTF-8''{encoded_name}"

    return FileResponse(
        doc.upload_path,
        media_type="application/pdf",
        headers={"Content-Disposition": content_disposition},
    )
