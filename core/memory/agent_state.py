"""Agent state persistence."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from core.agents.character_agent import CharacterAgent
from core.config import db_config
from core.models.schemas import AgentState


class AgentRecord(SQLModel, table=True):
    """Agent state record in database."""

    __tablename__ = "agents"

    id: Optional[int] = Field(default=None, primary_key=True)
    story_id: str = Field(index=True)
    agent_id: str = Field(unique=True, index=True)
    character_name: str
    active: bool = True
    state_json: str  # JSON serialized AgentState
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentStateStore:
    """Manages agent state persistence."""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the store.

        Args:
            db_url: Database URL (uses config default if not provided)
        """
        self.db_url = db_url or db_config.url
        self.engine = create_engine(self.db_url, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def save_agent(self, story_id: str, agent: CharacterAgent):
        """
        Persist agent state to database.

        Args:
            story_id: Story this agent belongs to
            agent: Character agent to save
        """
        state = AgentState(
            agent_id=agent.agent_id,
            dossier=agent.dossier,
            turn_memory=agent.memory_buffer.copy(),
            active=True,
            last_action_turn=len(agent.memory_buffer),
        )

        state_json = state.model_dump_json()

        with Session(self.engine) as session:
            # Check if exists
            statement = select(AgentRecord).where(AgentRecord.agent_id == agent.agent_id)
            existing = session.exec(statement).first()

            if existing:
                existing.state_json = state_json
                existing.updated_at = datetime.utcnow()
                existing.active = state.active
                session.add(existing)
            else:
                record = AgentRecord(
                    story_id=story_id,
                    agent_id=agent.agent_id,
                    character_name=agent.dossier.name,
                    state_json=state_json,
                )
                session.add(record)

            session.commit()

    def load_agent(self, agent_id: str) -> Optional[CharacterAgent]:
        """
        Restore agent from database.

        Args:
            agent_id: Agent identifier

        Returns:
            Restored character agent or None if not found
        """
        with Session(self.engine) as session:
            statement = select(AgentRecord).where(AgentRecord.agent_id == agent_id)
            record = session.exec(statement).first()

            if not record:
                return None

            state = AgentState.model_validate_json(record.state_json)

            agent = CharacterAgent(agent_id=state.agent_id, dossier=state.dossier)
            agent.memory_buffer = state.turn_memory.copy()

            return agent

    def list_agents(self, story_id: str) -> list[str]:
        """
        Get all agents for a story.

        Args:
            story_id: Story identifier

        Returns:
            List of agent IDs
        """
        with Session(self.engine) as session:
            statement = select(AgentRecord.agent_id).where(
                AgentRecord.story_id == story_id, AgentRecord.active == True
            )
            return list(session.exec(statement).all())

    def deactivate_agent(self, agent_id: str):
        """
        Deactivate an agent.

        Args:
            agent_id: Agent to deactivate
        """
        with Session(self.engine) as session:
            statement = select(AgentRecord).where(AgentRecord.agent_id == agent_id)
            record = session.exec(statement).first()

            if record:
                record.active = False
                record.updated_at = datetime.utcnow()
                session.add(record)
                session.commit()


# Global store instance
agent_store = AgentStateStore()
