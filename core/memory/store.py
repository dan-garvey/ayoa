"""SQLite-based story and event persistence."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from core.config import db_config


# Database models
class StoryRecord(SQLModel, table=True):
    """Story metadata record."""

    __tablename__ = "stories"

    id: Optional[int] = Field(default=None, primary_key=True)
    story_id: str = Field(unique=True, index=True)
    player_name: str
    genre: str
    tone: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    config_json: str  # JSON serialized StoryConfig
    outline_json: str  # JSON serialized StoryOutline


class EventRecord(SQLModel, table=True):
    """Story event/turn record."""

    __tablename__ = "events"

    id: Optional[int] = Field(default=None, primary_key=True)
    story_id: str = Field(index=True)
    turn_number: int
    event_type: str  # "opening", "turn", "scene_change"
    user_input: Optional[str] = None
    narrative: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data_json: str  # Full turn data as JSON


class SceneRecord(SQLModel, table=True):
    """Scene state record."""

    __tablename__ = "scenes"

    id: Optional[int] = Field(default=None, primary_key=True)
    story_id: str = Field(index=True)
    scene_id: str
    active: bool = True
    scene_json: str  # JSON serialized Scene
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StoryStore:
    """Manages story persistence in SQLite."""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the store.

        Args:
            db_url: Database URL (uses config default if not provided)
        """
        self.db_url = db_url or db_config.url
        self.engine = create_engine(self.db_url, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def save_story(
        self,
        story_id: str,
        player_name: str,
        genre: str,
        tone: str,
        config_json: str,
        outline_json: str,
    ):
        """Save or update story metadata."""
        with Session(self.engine) as session:
            # Check if exists
            statement = select(StoryRecord).where(StoryRecord.story_id == story_id)
            existing = session.exec(statement).first()

            if existing:
                existing.updated_at = datetime.utcnow()
                existing.config_json = config_json
                existing.outline_json = outline_json
                session.add(existing)
            else:
                record = StoryRecord(
                    story_id=story_id,
                    player_name=player_name,
                    genre=genre,
                    tone=tone,
                    config_json=config_json,
                    outline_json=outline_json,
                )
                session.add(record)

            session.commit()

    def load_story(self, story_id: str) -> Optional[StoryRecord]:
        """Load story metadata."""
        with Session(self.engine) as session:
            statement = select(StoryRecord).where(StoryRecord.story_id == story_id)
            return session.exec(statement).first()

    def save_event(
        self,
        story_id: str,
        turn_number: int,
        event_type: str,
        narrative: str,
        data_json: str,
        user_input: Optional[str] = None,
    ):
        """Save a story event/turn."""
        with Session(self.engine) as session:
            record = EventRecord(
                story_id=story_id,
                turn_number=turn_number,
                event_type=event_type,
                user_input=user_input,
                narrative=narrative,
                data_json=data_json,
            )
            session.add(record)
            session.commit()

    def load_events(self, story_id: str, limit: Optional[int] = None) -> list[EventRecord]:
        """Load story events."""
        with Session(self.engine) as session:
            statement = select(EventRecord).where(EventRecord.story_id == story_id).order_by(
                EventRecord.turn_number
            )
            if limit:
                statement = statement.limit(limit)
            return list(session.exec(statement).all())

    def save_scene(self, story_id: str, scene_id: str, scene_json: str):
        """Save scene state."""
        with Session(self.engine) as session:
            # Deactivate previous scenes
            statement = select(SceneRecord).where(
                SceneRecord.story_id == story_id, SceneRecord.active == True
            )
            for scene in session.exec(statement):
                scene.active = False
                session.add(scene)

            # Add new scene
            record = SceneRecord(
                story_id=story_id, scene_id=scene_id, scene_json=scene_json, active=True
            )
            session.add(record)
            session.commit()

    def load_active_scene(self, story_id: str) -> Optional[SceneRecord]:
        """Load the active scene for a story."""
        with Session(self.engine) as session:
            statement = select(SceneRecord).where(
                SceneRecord.story_id == story_id, SceneRecord.active == True
            )
            return session.exec(statement).first()


# Global store instance
story_store = StoryStore()
