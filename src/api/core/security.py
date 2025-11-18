from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.api.config import settings

# Password hashing context
# Configure bcrypt to truncate passwords at 72 bytes automatically
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__truncate_error=False,  # Don't raise error on long passwords
    bcrypt__ident="2b"  # Use 2b variant to avoid wrap-around bugs
)

# HTTP Bearer token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash

    Applies same 72-byte truncation as get_password_hash for consistency
    """
    # Truncate to 72 bytes to match hashing behavior
    password_bytes = plain_password.encode('utf-8')[:72]
    password_truncated = password_bytes.decode('utf-8', errors='ignore')
    return pwd_context.verify(password_truncated, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt

    Note: bcrypt has a 72-byte limit. We truncate to 72 bytes to avoid errors.
    This is a standard practice since password entropy beyond 72 bytes provides
    no additional security with bcrypt.
    """
    # Truncate to 72 bytes to handle bcrypt's limitation
    # This prevents errors with long passwords or multi-byte characters
    password_bytes = password.encode('utf-8')[:72]
    password_truncated = password_bytes.decode('utf-8', errors='ignore')
    return pwd_context.hash(password_truncated)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token

    Args:
        data: Data to encode in token (typically {"sub": user_id})
        expires_delta: Token expiration time (default: 30 minutes)

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire, "type": "access"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create JWT refresh token

    Args:
        data: Data to encode in token

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({"exp": expire, "type": "refresh"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Dependency to get current user ID from JWT token

    Usage:
        @app.get("/protected")
        async def protected_route(user_id: str = Depends(get_current_user_id)):
            ...
    """
    token = credentials.credentials
    payload = decode_token(token)

    # Verify token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    return user_id
