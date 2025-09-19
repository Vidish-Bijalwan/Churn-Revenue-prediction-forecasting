"""
ForeTel.AI Enterprise Authentication Service
OAuth 2.0, JWT, RBAC Implementation
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
import os
import redis
import hashlib
import secrets
from typing import Optional, List
import logging

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
REDIS_URL = os.getenv("REDIS_URL")

# Initialize FastAPI
app = FastAPI(
    title="ForeTel.AI Authentication Service",
    description="Enterprise-grade authentication with OAuth 2.0 and RBAC",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis Setup
redis_client = redis.from_url(REDIS_URL)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    roles = Column(Text)  # JSON string of roles
    permissions = Column(Text)  # JSON string of permissions

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String)
    user_agent = Column(String)
    is_active = Column(Boolean, default=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    action = Column(String, nullable=False)
    resource = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)
    user_agent = Column(String)
    details = Column(Text)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    roles: Optional[List[str]] = ["user"]

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    roles: List[str]
    permissions: List[str]

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token() -> str:
    return secrets.token_urlsafe(32)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

def get_current_user(username: str = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

def check_permission(required_permission: str):
    def permission_checker(current_user: User = Depends(get_current_user)):
        import json
        user_permissions = json.loads(current_user.permissions or "[]")
        if required_permission not in user_permissions and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return permission_checker

def log_audit_event(db: Session, user_id: int, action: str, resource: str = None, 
                   ip_address: str = None, user_agent: str = None, details: str = None):
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        resource=resource,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details
    )
    db.add(audit_log)
    db.commit()

# Routes
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "auth", "timestamp": datetime.utcnow()}

@app.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Create new user
    import json
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        roles=json.dumps(user_data.roles),
        permissions=json.dumps(["read_own_data", "update_own_profile"])
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Log audit event
    log_audit_event(db, db_user.id, "USER_REGISTERED", "users")
    
    return UserResponse(
        id=db_user.id,
        email=db_user.email,
        username=db_user.username,
        full_name=db_user.full_name,
        is_active=db_user.is_active,
        roles=json.loads(db_user.roles),
        permissions=json.loads(db_user.permissions)
    )

@app.post("/login", response_model=Token)
async def login(login_data: UserLogin, db: Session = Depends(get_db)):
    # Get user
    user = db.query(User).filter(User.username == login_data.username).first()
    
    if not user or not verify_password(login_data.password, user.hashed_password):
        # Log failed attempt
        if user:
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
            db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account is temporarily locked due to multiple failed login attempts"
        )
    
    # Reset failed attempts and update last login
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create tokens
    access_token = create_access_token({"sub": user.username, "user_id": user.id})
    refresh_token = create_refresh_token()
    
    # Store session
    session_id = hashlib.sha256(f"{user.id}{datetime.utcnow()}".encode()).hexdigest()
    session = Session(
        id=session_id,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=30),
        ip_address="0.0.0.0",  # Get from request
        user_agent="Unknown"   # Get from request
    )
    db.add(session)
    db.commit()
    
    # Cache in Redis
    redis_client.setex(f"session:{session_id}", 86400, user.username)
    redis_client.setex(f"refresh_token:{refresh_token}", 2592000, user.username)
    
    # Log audit event
    log_audit_event(db, user.id, "USER_LOGIN", "sessions")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        refresh_token=refresh_token
    )

@app.post("/logout")
async def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Invalidate sessions
    sessions = db.query(Session).filter(Session.user_id == current_user.id, Session.is_active == True).all()
    for session in sessions:
        session.is_active = False
        redis_client.delete(f"session:{session.id}")
    
    db.commit()
    
    # Log audit event
    log_audit_event(db, current_user.id, "USER_LOGOUT", "sessions")
    
    return {"message": "Successfully logged out"}

@app.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    import json
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        roles=json.loads(current_user.roles or "[]"),
        permissions=json.loads(current_user.permissions or "[]")
    )

@app.post("/refresh-token", response_model=Token)
async def refresh_access_token(refresh_token: str, db: Session = Depends(get_db)):
    username = redis_client.get(f"refresh_token:{refresh_token}")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = db.query(User).filter(User.username == username.decode()).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new access token
    access_token = create_access_token({"sub": user.username, "user_id": user.id})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        refresh_token=refresh_token
    )

@app.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(check_permission("manage_users")),
    db: Session = Depends(get_db)
):
    users = db.query(User).all()
    import json
    return [
        UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            roles=json.loads(user.roles or "[]"),
            permissions=json.loads(user.permissions or "[]")
        )
        for user in users
    ]

@app.get("/audit-logs")
async def get_audit_logs(
    current_user: User = Depends(check_permission("view_audit_logs")),
    db: Session = Depends(get_db),
    limit: int = 100
):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return logs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
