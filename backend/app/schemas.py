
from pydantic import BaseModel, Field
from pydantic import BaseModel, EmailStr

class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=150)
    password: str = Field(min_length=8, max_length=128)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str