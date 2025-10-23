# Multi-Agent Narrative Game

A Python-based interactive narrative game engine where multiple AI-powered character agents maintain persistent identities and interact dynamically with the player. Each character has their own goals, secrets, and personalities, creating a living story world.

## Features

- **Persistent Character Agents**: Each major character is a separate agent with their own memory, goals, and personality
- **Information Routing**: Director intelligently routes information based on scene presence and character abilities
- **Dynamic Narrative**: Storyteller weaves character actions into coherent prose
- **State Persistence**: Full save/load system with SQLite backend
- **Flexible Architecture**: Single vLLM model serves all roles with separate contexts
- **CLI-First**: Rich terminal interface with Typer

## Architecture

The system uses three core AI roles:

1. **Storyteller**: Generates story outlines, creates scenes, and composes narrative prose
2. **Director**: Routes information to characters and validates their moves
3. **Character Agents**: Each major character maintains their own identity, memories, and decision-making

### Key Components

```
core/
├── roles/          # AI role implementations (Storyteller, Director, Character)
├── agents/         # Character agent system with persistence
├── engine/         # Game orchestration and information routing
├── memory/         # SQLite persistence and state management
├── models/         # Pydantic schemas for all data structures
├── api/            # FastAPI server
└── cli.py          # Typer CLI interface
```

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **vLLM Server** running with OpenAI-compatible API

### Setup vLLM Server

```bash
# Install vLLM
pip install vllm

# Start server with a small model for testing
vllm serve Qwen/Qwen1.5-0.5B --port 8000
```

For better results, use a larger model like:
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `Qwen/Qwen2.5-7B-Instruct`

### Install the Game

```bash
# Clone or download the project
cd core

# Create virtual environment and install
make setup

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

Required settings:
```env
OPENAI_BASE_URL=http://localhost:8000/v1
MODEL_NAME=Qwen/Qwen1.5-0.5B
MAX_CONTEXT_TOKENS=8192
```

## Usage

### Create a New Story

```bash
story create
```

This launches an interactive wizard that asks about:
- Your character (name, background, traits, motivations, appearance, skills)
- Story preferences (genre, tone, themes, length, content boundaries)

The system generates a story outline with major characters and returns a `story_id`.

### Start Playing

```bash
# Start the story (spawns character agents and generates opening)
story start <story_id>

# Continue playing interactively
story continue <story_id>

# Or provide a single action
story continue <story_id> --input "I carefully examine the mysterious letter"
```

### Game Commands

During gameplay:
- Type your actions naturally: `I approach Lord Ashford and bow respectfully`
- Use meta commands:
  - `/scene` - Describe current scene
  - `/cast` - List active characters
  - `/quit` - Save and exit

### Inspect Story State

```bash
# View current scene
story inspect <story_id> --what scene

# View active characters
story inspect <story_id> --what cast

# View story outline
story inspect <story_id> --what outline

# View character dossiers (full details)
story inspect <story_id> --what dossiers

# View event log
story inspect <story_id> --what eventlog
```

### Other Commands

```bash
# List all saved stories
story list-stories

# Show configuration
story config

# Save story (automatic after each turn, but manual option available)
story save <story_id>

# Load story
story load <story_id>
```

## Example Session

```bash
$ export OPENAI_BASE_URL=http://localhost:8000/v1
$ story create

CHARACTER CREATION
==================

What is your character's name? > Eleanor Blackwood

Describe your character's background (2-3 sentences)
> A former court physician who discovered a conspiracy. Now seeking evidence
  while maintaining their cover.

List 3 personality traits (comma-separated)
> analytical, cautious, compassionate

What are your character's main motivations or goals?
> uncover the truth, protect the innocent

Briefly describe your character's appearance
> Tall with silver-streaked dark hair, sharp green eyes

Any special skills or abilities?
> medicine, diplomacy, observation

STORY PREFERENCES
=================

What genre interests you? > Regency-fantasy intrigue

What tone do you prefer? > witty

Any themes you'd like to explore? > betrayal, redemption

Story length: short (2-3 hours), medium (5-6 hours), or long (8+ hours)? > short

Generating your story...

Created story: story_7a8b9c

Premise: Eleanor Blackwood, once the king's trusted physician, has uncovered
evidence of a conspiracy at the highest levels of court...

Major Characters:
  - Lord Ashford (antagonist): Ambitious noble with dangerous secrets
  - Marcus Webb (ally): Fellow physician and confidant
  - Lady Vivienne (rival): Sharp-tongued courtier with her own agenda

Start your adventure now? > yes

============================================================
The morning mist clings to the cobblestones of Ravenshollow
as Eleanor Blackwood examines the ledger one more time. The
numbers don't lie—someone has been siphoning funds from the
royal treasury, and the trail leads directly to the palace...
============================================================

> I hide the ledger in my medical bag and head to the palace

Processing...

============================================================
Eleanor slipped the incriminating ledger between packets of
herbs in her medical bag, her fingers steady despite the
weight of what she'd discovered. As she made her way through
the palace corridors, Marcus fell into step beside her.

"You found something," he said quietly, not quite a question.

Eleanor met his concerned gaze. "More than I wished to."

From an alcove ahead, Lady Vivienne's laugh rang out, sharp
and knowing. She emerged with Lord Ashford at her side, both
watching Eleanor with predatory interest.

"Dr. Blackwood," Ashford said smoothly, "how fortunate. I was
just telling Lady Vivienne about the king's improving health.
You'll confirm this, I trust?"
============================================================

> I maintain my composure and reply professionally about the king's condition

[Game continues...]
```

## Configuration

### Environment Variables

- `OPENAI_BASE_URL`: vLLM server URL (required)
- `OPENAI_API_KEY`: API key (optional, defaults to "EMPTY")
- `MODEL_NAME`: Model to use for all roles
- `MAX_CONTEXT_TOKENS`: Maximum context window
- `MAX_ACTIVE_CHARACTERS_PER_TURN`: How many characters to query concurrently (default: 4)
- `RNG_SEED`: Random seed for deterministic behavior (default: 1337)

### Temperature Settings

- `DIRECTOR_TEMPERATURE`: Default 0.2 (precise, logical routing)
- `STORYTELLER_TEMPERATURE`: Default 0.7 (creative narrative)
- `CHARACTER_DEFAULT_TEMPERATURE`: Default 0.7 (can be overridden per character)

### Database

- `DATABASE_URL`: SQLite database location (default: `sqlite:///./core.db`)

## API Server

Run the FastAPI server:

```bash
make server
# Or: uvicorn core.api.server:app --reload --port 8081
```

Endpoints:
- `POST /story/create` - Create new story
- `POST /story/start` - Start created story
- `POST /story/turn` - Process a turn
- `GET /story/agents/{story_id}` - List agents
- `GET /story/inspect/{story_id}` - Inspect story state
- `GET /health` - Health check

## Development

### Running Tests

```bash
# Run all tests with coverage
make test

# Quick test run (stop on first failure)
make test-quick

# Run specific test file
pytest tests/test_character_agents.py -v
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint
```

### Project Structure

```
core/
├── __init__.py
├── config.py              # Configuration management
├── llm_client.py          # vLLM client
├── cli.py                 # CLI interface
├── prompts/               # System prompts for each role
│   ├── director.txt
│   ├── storyteller.txt
│   ├── character.txt
│   └── character_creation.txt
├── models/
│   └── schemas.py         # All Pydantic models
├── roles/
│   ├── storyteller.py     # Storyteller implementation
│   ├── director.py        # Director implementation
│   └── character.py       # Character role base
├── agents/
│   ├── character_agent.py # Character agent with memory
│   └── manager.py         # Agent lifecycle management
├── engine/
│   ├── orchestrator.py    # Main game loop
│   ├── information_router.py
│   └── rng.py            # Seeded RNG
├── memory/
│   ├── store.py          # SQLite persistence
│   ├── agent_state.py    # Agent state management
│   └── vector.py         # Vector memory (optional)
└── api/
    └── server.py         # FastAPI server
```

## How It Works

### Story Creation (Two-Phase)

**Phase 1: Outline Generation**
1. Player creates character and sets preferences
2. Storyteller generates story outline with major characters
3. System returns `story_id` and outline

**Phase 2: Game Initialization**
1. Director spawns persistent agent for each major character
2. Storyteller creates opening scene (without major characters present yet)
3. Opening narrative is composed

### Turn Processing Loop

For each player action:

1. **Information Routing**: Director analyzes scene and decides which characters should be aware of events
   - Characters present in scene get full information
   - Nearby characters may overhear or glimpse things (partial attention)
   - Remote characters typically receive nothing (unless special abilities)

2. **Character Responses**: Each informed character agent:
   - Processes information through their own perspective
   - Decides whether to act based on goals and personality
   - May choose to observe silently
   - Maintains separate memory and context

3. **Move Validation**: Director reviews all character responses:
   - Validates moves for consistency and possibility
   - Resolves conflicts between moves
   - Identifies needed NPC reactions

4. **Narrative Composition**: Storyteller weaves together:
   - Player action
   - Accepted character moves (preserving exact dialogue)
   - NPC reactions
   - Environmental details
   - Result: 200-500 words of prose

5. **State Update**:
   - All agents update their memories
   - Scene state updated if needed
   - Everything persisted to disk

### Character Agent Memory

Each character agent maintains:
- **Dossier**: Identity, goals, secrets, beliefs, relationships, emotional state
- **Style Card**: Voice, speech patterns, catchphrases, temperature override
- **Turn Memory**: Last 20 turns of context
- **Long-term Memories**: Key events stored permanently

Agents can:
- Update beliefs based on observations
- Track relationships with other characters
- Choose when to act vs observe
- Keep secrets until forced to reveal them

## Customization

### Adding Custom Prompts

Edit files in `core/prompts/` to customize AI behavior:
- `storyteller.txt` - Narrative style and world-building
- `director.txt` - Routing logic and move validation
- `character.txt` - Character response generation

### Adjusting Agent Behavior

Modify character style cards to control:
- Temperature (0.5-0.9 for more/less variability)
- Speech patterns and voice
- Taboo topics
- Catchphrases

### Extending the System

Future enhancements could include:
- **Advanced Memory**: Semantic search with embeddings
- **Context Management**: Automatic summarization when context fills
- **Dynamic Spawning**: Create new characters mid-story
- **Interrupt System**: Characters can interrupt each other
- **Relationship Graphs**: Visualize character relationships
- **Multi-timeline**: Handle parallel scenes

## Performance Tips

1. **Model Selection**: Larger models (7B+) produce better narrative quality
2. **Concurrent Characters**: Adjust `MAX_ACTIVE_CHARACTERS_PER_TURN` based on your hardware
3. **Context Window**: Use models with large context windows (8K+ recommended)
4. **Batch Processing**: The system batches character queries to maximize throughput

## Troubleshooting

### vLLM Connection Issues
- Ensure vLLM server is running: `curl http://localhost:8000/v1/models`
- Check `OPENAI_BASE_URL` in `.env`

### Slow Response Times
- Use a faster model or reduce `MAX_ACTIVE_CHARACTERS_PER_TURN`
- Check vLLM server logs for bottlenecks

### JSON Parsing Errors
- Some smaller models struggle with JSON output
- Try a larger, instruction-tuned model
- Adjust temperature down for more consistent JSON

### Out of Memory
- Reduce `MAX_CONTEXT_TOKENS`
- Implement memory truncation more aggressively
- Use a model with lower memory requirements

## License

This project is provided as-is for educational and entertainment purposes.

## Contributing

Contributions welcome! Key areas:
- Additional test coverage
- Memory management optimizations
- UI improvements
- Documentation

## Acknowledgments

Built with:
- FastAPI for the API server
- Typer for the CLI
- SQLModel for database ORM
- Pydantic for data validation
- Rich for terminal formatting
- vLLM for high-performance LLM inference
