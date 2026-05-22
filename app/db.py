"""
db.py - Database setup using SQLAlchemy
Supports SQLite (dev) and PostgreSQL (Render production)
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sentifi.db")

# Render gives postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SentimentRecord(Base):
    __tablename__ = "sentiment_records"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), index=True, nullable=False)
    source = Column(String(20), nullable=False)        # 'reddit' or 'twitter'
    text = Column(Text, nullable=False)
    score = Column(Float, nullable=False)              # -1.0 to 1.0
    label = Column(String(10), nullable=False)         # positive/negative/neutral
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalRecord(Base):
    __tablename__ = "signal_records"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), index=True, nullable=False)
    signal = Column(String(10), nullable=False)        # BUY / SELL / HOLD
    avg_score = Column(Float, nullable=False)
    weighted_score = Column(Float, nullable=False)
    post_count = Column(Integer, nullable=False)
    reddit_score = Column(Float, nullable=True)
    twitter_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
