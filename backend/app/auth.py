
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import User
from app.schemas import RegisterRequest, LoginRequest
from app.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register")
def register_user(
    data: RegisterRequest,
    db: Session = Depends(get_db)
):
    # Check whether the email is already registered
    existing_user = db.query(User).filter(
        User.email == data.email
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email is already registered"
        )

    # Hash the password before storing it
    new_user = User(
        full_name=data.full_name,
        email=data.email,
        hashed_password=hash_password(data.password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User registered successfully",
        "user": {
            "id": new_user.id,
            "full_name": new_user.full_name,
            "email": new_user.email
        }
    
    }

@router.post("/login")
def login_user(
    data: LoginRequest,
    db: Session = Depends(get_db)
):
    # Find user by email
    user = db.query(User).filter(
        User.email == data.email
    ).first()

    # Check email and password
    if not user or not verify_password(
        data.password,
        user.hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # Create JWT access token
    access_token = create_access_token(
        data={"sub": str(user.id)}
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email
        }
    }
