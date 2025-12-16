from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import get_current_user, get_db
from app.repositories.users import UsersRepository
from app.schemas.auth import Token
from app.schemas.user import UserCreate, UserPublic
from app.services.auth import AuthService

router = APIRouter()


@router.post("/signup", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserCreate, db=Depends(get_db)) -> UserPublic:
    repo = UsersRepository(db)
    existing = await repo.get_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    auth = AuthService()
    user = await repo.create_user(email=payload.email, password_hash=auth.hash_password(payload.password))
    return UserPublic.model_validate(user)


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)) -> Token:
    repo = UsersRepository(db)
    user = await repo.get_by_email(form.username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    auth = AuthService()
    if not auth.verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = auth.create_access_token(subject=str(user.id))
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserPublic)
async def me(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return current_user
