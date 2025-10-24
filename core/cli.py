"""Typer-based CLI for the multi-agent narrative game."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from core.engine.orchestrator import orchestrator
from core.models.schemas import PlayerCharacter, StoryPreferences

app = typer.Typer(help="Multi-Agent Narrative Game")
console = Console()


def run_async(coro):
    """Helper to run async functions in sync context."""
    return asyncio.run(coro)


@app.command()
def create():
    """Interactive character and story creation."""
    console.print(Panel.fit("CHARACTER CREATION", style="bold blue"))

    # Character questions
    name = Prompt.ask("What is your character's name?")

    console.print("\n[dim]Describe your character's background (2-3 sentences)[/dim]")
    background = Prompt.ask("Background")

    console.print("\n[dim]List 3 personality traits (comma-separated)[/dim]")
    traits_input = Prompt.ask("Traits", default="brave, clever, determined")
    traits = [t.strip() for t in traits_input.split(",")]

    console.print("\n[dim]What are your character's main motivations or goals?[/dim]")
    motivations_input = Prompt.ask("Motivations", default="seek truth, protect loved ones")
    motivations = [m.strip() for m in motivations_input.split(",")]

    appearance = Prompt.ask("\nBriefly describe your character's appearance")

    skills_input = Prompt.ask(
        "\nAny special skills or abilities? (comma-separated, optional)", default=""
    )
    skills = [s.strip() for s in skills_input.split(",") if s.strip()]

    player_char = PlayerCharacter(
        name=name,
        background=background,
        traits=traits,
        motivations=motivations,
        appearance=appearance,
        skills=skills,
    )

    # Story preferences
    console.print("\n")
    console.print(Panel.fit("STORY PREFERENCES", style="bold blue"))

    genre = Prompt.ask(
        "\nWhat genre interests you?",
        default="Political intrigue",
    )

    tone = Prompt.ask(
        "What tone do you prefer? (e.g., dark, witty, epic, intimate)", default="witty"
    )

    themes_input = Prompt.ask(
        "Any themes you'd like to explore? (comma-separated, optional)", default=""
    )
    themes = [t.strip() for t in themes_input.split(",") if t.strip()]

    length = Prompt.ask(
        "Story length? (short/medium/long)", default="short", choices=["short", "medium", "long"]
    )

    content_boundaries_input = Prompt.ask(
        "Any content you'd prefer to avoid? (comma-separated, optional)", default=""
    )
    content_boundaries = [c.strip() for c in content_boundaries_input.split(",") if c.strip()]

    preferences = StoryPreferences(
        genre=genre,
        tone=tone,
        themes=themes,
        length=length,  # type: ignore
        content_boundaries=content_boundaries,
    )

    # Create story
    console.print("\n[bold yellow]Generating your story...[/bold yellow]")

    from core.models.schemas import StoryConfig

    config = StoryConfig(player_character=player_char, preferences=preferences)

    story_id, outline = run_async(orchestrator.create_story(config))

    console.print(f"\n[bold green]Created story: {story_id}[/bold green]")
    console.print(f"\n[bold]Premise:[/bold] {outline.premise}")
    console.print(f"\n[bold]Major Characters:[/bold]")
    for char in outline.major_characters:
        console.print(f"  - {char.name} ({char.role}): {char.description}")

    # Auto-start
    if Confirm.ask("\nStart your adventure now?"):
        start(story_id)
    else:
        console.print(f"\n[dim]To start later, run: story start {story_id}[/dim]")


@app.command()
def start(
    story_id: str,
    max_history: Optional[int] = typer.Option(
        None,
        "--max-history",
        "-m",
        help="Maximum narrative turns to keep in Storyteller memory (default: 20)",
    ),
):
    """Start a created story."""
    console.print(f"[bold yellow]Initializing story {story_id}...[/bold yellow]")

    try:
        # Set max history if provided
        if max_history is not None:
            orchestrator.storyteller.set_max_history_turns(max_history)
            console.print(f"[dim]Storyteller memory limit: {max_history} turns[/dim]")

        opening = run_async(orchestrator.start_story(story_id))

        console.print("\n" + "=" * 80 + "\n")
        console.print(Panel(opening.narrative, title="Opening Scene", border_style="blue"))
        console.print("\n" + "=" * 80 + "\n")

        console.print(f"[dim]Story started! Use: story continue-story {story_id}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def continue_story(
    story_id: str, input_text: Optional[str] = typer.Option(None, "--input", "-i", help="Your action")
):
    """Continue an active story."""
    try:
        if input_text is None:
            console.print(
                f"[bold]Story {story_id}[/bold] - What do you do? (Type /quit to exit)\n"
            )

        while True:
            if input_text is None:
                user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
            else:
                user_input = input_text

            if user_input.lower() in ["/quit", "/exit", "/q"]:
                console.print("[dim]Saving and exiting...[/dim]")
                break

            # Process turn
            console.print("\n[dim]Processing...[/dim]\n")
            output = run_async(orchestrator.process_turn(story_id, user_input))

            console.print("=" * 80 + "\n")
            console.print(output.narrative)
            console.print("\n" + "=" * 80 + "\n")

            # If input was provided as option, exit after one turn
            if input_text is not None:
                break

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def save(story_id: str):
    """Save a story."""
    try:
        orchestrator._save_story_state(story_id)
        console.print(f"[bold green]Story {story_id} saved successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def load(story_id: str):
    """Load a story."""
    try:
        orchestrator._load_story_state(story_id)
        console.print(f"[bold green]Story {story_id} loaded successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def inspect(
    story_id: str,
    what: str = typer.Option(
        "scene",
        help="What to inspect: scene, cast, agents, outline, eventlog, dossiers",
    ),
):
    """Inspect various aspects of a story."""
    try:
        # Load story if needed
        if orchestrator.current_story_id != story_id:
            orchestrator._load_story_state(story_id)

        if what == "scene":
            if orchestrator.current_scene:
                scene = orchestrator.current_scene
                console.print(Panel.fit(f"SCENE: {scene.scene_id}", style="bold"))
                console.print(f"[bold]Where:[/bold] {scene.where}")
                console.print(f"[bold]When:[/bold] {scene.when}")
                console.print(f"[bold]Atmosphere:[/bold] {scene.atmosphere}")
                console.print(f"[bold]Present:[/bold] {', '.join(scene.present_characters)}")
                if scene.nearby_characters:
                    console.print(f"[bold]Nearby:[/bold] {', '.join(scene.nearby_characters)}")
            else:
                console.print("[dim]No active scene[/dim]")

        elif what == "cast":
            agents = orchestrator.agent_manager.list_agents()
            console.print(Panel.fit("ACTIVE CHARACTERS", style="bold"))
            for agent_id, state in agents.items():
                console.print(
                    f"- {state.dossier.name} ({state.dossier.character_concept.role}) [{agent_id}]"
                )

        elif what == "outline":
            if orchestrator.current_outline:
                outline = orchestrator.current_outline
                console.print(Panel.fit("STORY OUTLINE", style="bold"))
                console.print(f"[bold]Premise:[/bold] {outline.premise}\n")
                console.print("[bold]Acts:[/bold]")
                for i, act in enumerate(outline.acts, 1):
                    console.print(f"  {i}. {act}")
            else:
                console.print("[dim]No outline available[/dim]")

        elif what == "eventlog":
            console.print(Panel.fit("EVENT LOG", style="bold"))
            for event in orchestrator.turn_history[-10:]:  # Last 10 events
                console.print(f"Turn {event.get('turn', '?')}: {event.get('summary', 'Event')}")

        elif what == "dossiers":
            agents = orchestrator.agent_manager.list_agents()
            for agent_id, state in agents.items():
                dossier = state.dossier
                console.print(Panel.fit(f"DOSSIER: {dossier.name}", style="bold"))
                console.print(f"[bold]Role:[/bold] {dossier.character_concept.role}")
                console.print(
                    f"[bold]Personality:[/bold] {', '.join(dossier.character_concept.personality)}"
                )
                console.print(f"[bold]Goals:[/bold] {', '.join(dossier.current_goals)}")
                console.print(f"[bold]Emotional State:[/bold] {dossier.emotional_state}")
                if dossier.relationships:
                    console.print("[bold]Relationships:[/bold]")
                    for name, stance in dossier.relationships.items():
                        console.print(f"  - {name}: {stance}")
                console.print()

        else:
            console.print(f"[bold red]Unknown inspection type: {what}[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def config():
    """Print active configuration."""
    from core.config import engine_config, llm_config

    console.print(Panel.fit("CONFIGURATION", style="bold blue"))
    console.print(f"[bold]Model:[/bold] {llm_config.model_name}")
    console.print(f"[bold]Base URL:[/bold] {llm_config.base_url}")
    console.print(f"[bold]Max Context:[/bold] {llm_config.max_context_tokens}")
    console.print(
        f"[bold]Max Active Characters:[/bold] {engine_config.max_active_characters_per_turn}"
    )
    console.print(f"[bold]RNG Seed:[/bold] {engine_config.rng_seed}")
    console.print("\n[bold]Temperatures:[/bold]")
    console.print(f"  Director: {engine_config.director_params.temperature}")
    console.print(f"  Storyteller: {engine_config.storyteller_params.temperature}")
    console.print(f"  Character Default: {engine_config.character_default_params.temperature}")


@app.command()
def list_stories():
    """List all saved stories."""
    saves_dir = Path("./saves")
    if not saves_dir.exists():
        console.print("[dim]No saved stories found[/dim]")
        return

    stories = list(saves_dir.glob("*.json"))
    if not stories:
        console.print("[dim]No saved stories found[/dim]")
        return

    console.print(Panel.fit("SAVED STORIES", style="bold blue"))
    for story_file in stories:
        story_id = story_file.stem
        console.print(f"- {story_id}")


if __name__ == "__main__":
    app()
