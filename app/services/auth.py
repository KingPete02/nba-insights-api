from __future__ import annotations

from passlib.context import CryptContext

from app.core.security import create_access_token

_pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
)


class AuthService:
    def hash_password(self, password: str) -> str:
        return _pwd_context.hash(password)

    def verify_password(self, plain_password: str, password_hash: str) -> bool:
        return _pwd_context.verify(plain_password, password_hash)

    def create_access_token(self, subject: str) -> str:
        return create_access_token(subject=subject)
