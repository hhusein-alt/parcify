from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class UserProfile(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    preferences: Optional[dict] = None

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Optional[dict] = None

class SearchHistory(BaseModel):
    id: str
    user_id: str
    query: str
    timestamp: datetime
    results_count: int
    filters: Optional[dict] = None

class SavedSearch(BaseModel):
    id: str
    user_id: str
    name: str
    query: str
    filters: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

class ExportHistory(BaseModel):
    id: str
    user_id: str
    format: str
    filename: str
    timestamp: datetime
    data_size: int
    status: str 