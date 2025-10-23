"""Information routing for filtering character perception."""

from core.models.schemas import AgentState, RoutingDecision, Scene
from core.roles.director import Director


class InformationRouter:
    """Routes information to characters based on scene position and abilities."""

    def __init__(self, director: Director):
        """
        Initialize the router.

        Args:
            director: Director instance for making routing decisions
        """
        self.director = director

    async def route_information(
        self,
        scene: Scene,
        user_input: str,
        agent_registry: dict[str, AgentState],
        recent_events: list[str],
    ) -> list[RoutingDecision]:
        """
        Director decides who learns what.

        Args:
            scene: Current scene
            user_input: Player's action
            agent_registry: Available agents
            recent_events: Recent story events

        Returns:
            Routing decisions for each character
        """
        decisions = await self.director.route_information(
            scene=scene, user_input=user_input, agents=agent_registry, recent_history=recent_events
        )

        return decisions
