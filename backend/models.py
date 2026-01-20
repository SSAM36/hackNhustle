from datetime import datetime
from bson import ObjectId

class BaseModel:
    """Base model with common fields"""
    def __init__(self):
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self):
        """Convert model to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

class User(BaseModel):
    """User model"""
    def __init__(self, username, email, password_hash, role='user'):
        super().__init__()
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self.is_active = True

class Role(BaseModel):
    """Role model"""
    def __init__(self, name, permissions=None):
        super().__init__()
        self.name = name
        self.permissions = permissions or []
        self.description = ""

class Video(BaseModel):
    """Video model"""
    def __init__(self, title, url, duration=0, user_id=None, category=None):
        super().__init__()
        self.title = title
        self.url = url
        self.duration = duration  # in seconds
        self.user_id = user_id
        self.category = category  # alphabet, numbers, words, etc.
        self.thumbnail_url = ""
        self.description = ""
        self.tags = []
        self.is_public = True

class PracticeSession(BaseModel):
    """Practice session model"""
    def __init__(self, user_id, video_id, session_type='practice'):
        super().__init__()
        self.user_id = user_id
        self.video_id = video_id
        self.session_type = session_type  # practice, test, review
        self.start_time = datetime.utcnow()
        self.end_time = None
        self.duration = 0  # in seconds
        self.score = 0
        self.completed = False
        self.notes = ""

class AnalyticsEvent(BaseModel):
    """Analytics event model"""
    def __init__(self, user_id, event_type, event_data=None):
        super().__init__()
        self.user_id = user_id
        self.event_type = event_type  # video_view, practice_start, practice_end, etc.
        self.event_data = event_data or {}
        self.session_id = None
        self.timestamp = datetime.utcnow()