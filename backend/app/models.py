
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    ForeignKey,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


# =========================================
# USER MODEL
# =========================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    # Relationship with prediction history
    predictions = relationship(
        "PredictionHistory",
        back_populates="user",
        cascade="all, delete-orphan"
    )


# =========================================
# PREDICTION HISTORY MODEL
# =========================================

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    # User who performed the assessment
    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # Disease assessment type
    disease = Column(
        String(50),
        nullable=False
    )

    # Model prediction: 0 or 1
    prediction = Column(
        Integer,
        nullable=False
    )

    # Estimated model probability in percentage
    estimated_probability = Column(
        Float,
        nullable=False
    )

    # Prediction message
    message = Column(
        String(255),
        nullable=True
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationship with User
    user = relationship(
        "User",
        back_populates="predictions"
    )
