"""
Database Schemas for Gamma

Each Pydantic model maps to a MongoDB collection (lowercased class name).
- User -> user
- Document -> document
- Summary -> summary
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal
from datetime import datetime


class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Unique email")
    password_hash: str = Field(..., description="BCrypt hashed password")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    plan: Literal["free", "pro", "enterprise"] = "free"
    is_active: bool = True


class Document(BaseModel):
    owner_id: str = Field(..., description="User id (stringified ObjectId)")
    filename: str
    mime_type: str
    size: int
    text: Optional[str] = None
    visibility: Literal["private", "shared"] = "private"
    share_id: Optional[str] = None
    tags: List[str] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Summary(BaseModel):
    owner_id: str
    document_id: str
    style: Literal["concise", "detailed", "bullets", "legal", "executive"] = "concise"
    prompt: Optional[str] = None
    content: str
    model: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    shared: bool = False
    share_id: Optional[str] = None
