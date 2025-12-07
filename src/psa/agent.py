"""
PSA v3 PlasticAgent - High-Level Agent Wrapper
----------------------------------------------
A simple, user-friendly wrapper around LiquidAgent for conversational use.

Features:
- Simple .chat() method for conversations
- Automatic state management
- Skill loading and composition
- Conversation history tracking

Usage:
    from psa.agent import PlasticAgent

    agent = PlasticAgent("path/to/model.pth")
    agent.load_skill("medical", weight=0.8)
    response = agent.chat("What are the symptoms of diabetes?")
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    perplexity: Optional[float] = None


@dataclass
class Response:
    """Response from the agent."""
    text: str
    state_updated: bool = True
    skills_active: List[str] = field(default_factory=list)
    generation_time_ms: float = 0.0


class PlasticAgent:
    """
    High-level conversational agent with State Algebra capabilities.

    Wraps LiquidAgent for easy use:
    - Manages conversation state automatically
    - Supports skill loading and composition
    - Tracks conversation history
    """

    def __init__(
        self,
        model_path: str,
        strategy: str = "cuda fp16",
        skills_dir: Optional[str] = None,
    ):
        """
        Initialize PlasticAgent.

        Args:
            model_path: Path to RWKV model (.pth)
            strategy: Inference strategy ('cuda fp16', 'cpu fp32', etc.)
            skills_dir: Directory for skill files (default: ~/.psa/skills)
        """
        from psa.kernel import LiquidAgent

        self.kernel = LiquidAgent(model_path, strategy=strategy)

        # Skills management
        self.skills_dir = Path(skills_dir) if skills_dir else Path.home() / ".psa" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # State management
        self.current_state: Optional[List[torch.Tensor]] = None
        self.base_state: Optional[List[torch.Tensor]] = None  # Before any learning
        self.loaded_skills: Dict[str, float] = {}  # skill_name -> weight

        # Conversation tracking
        self.conversation_history: List[ConversationTurn] = []
        self.system_prompt: Optional[str] = None

        # Generation settings
        self.temperature: float = 0.8
        self.max_tokens: int = 256
        self.top_p: float = 0.85

    def set_system_prompt(self, prompt: str):
        """
        Set system prompt that influences all responses.

        The system prompt is processed into state at the start of conversation.
        """
        self.system_prompt = prompt

        # Process system prompt into state
        self.current_state = self.kernel.learn_stream(
            f"System: {prompt}",
            self.current_state
        )

    def load_skill(self, skill_name: str, weight: float = 1.0) -> bool:
        """
        Load a skill and apply it to current state.

        Args:
            skill_name: Name of skill file (without .psa extension)
            weight: Weight for the skill (0.0 to 1.0+)

        Returns:
            True if skill loaded successfully
        """
        skill_path = self.skills_dir / f"{skill_name}.psa"

        if not skill_path.exists():
            return False

        # Load skill delta
        skill_delta = self.kernel.load_state(str(skill_path))

        # Apply to current state
        self.current_state = self.kernel.apply_delta(
            self.current_state,
            skill_delta,
            weight=weight
        )

        self.loaded_skills[skill_name] = weight
        return True

    def load_skills(self, skills: Dict[str, float]) -> int:
        """
        Load multiple skills with weights.

        Args:
            skills: Dict mapping skill names to weights

        Returns:
            Number of skills successfully loaded
        """
        loaded = 0
        for skill_name, weight in skills.items():
            if self.load_skill(skill_name, weight):
                loaded += 1
        return loaded

    def unload_skill(self, skill_name: str) -> bool:
        """
        Unload a skill by subtracting it from state.

        Args:
            skill_name: Name of skill to remove

        Returns:
            True if skill was unloaded
        """
        if skill_name not in self.loaded_skills:
            return False

        skill_path = self.skills_dir / f"{skill_name}.psa"
        if not skill_path.exists():
            return False

        weight = self.loaded_skills[skill_name]
        skill_delta = self.kernel.load_state(str(skill_path))

        # Subtract skill (apply with negative weight)
        self.current_state = self.kernel.apply_delta(
            self.current_state,
            skill_delta,
            weight=-weight
        )

        del self.loaded_skills[skill_name]
        return True

    def chat(
        self,
        message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Response:
        """
        Send a message and get a response.

        This is the main interface for conversation. State is automatically
        updated after each turn.

        Args:
            message: User message
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            Response object with generated text
        """
        start_time = time.time()

        # Record user turn
        self.conversation_history.append(ConversationTurn(
            role="user",
            content=message,
        ))

        # Generate response
        response_text, new_state = self.kernel.infer(
            ctx=message,
            state=self.current_state,
            token_count=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            top_p=self.top_p,
        )

        # Update state
        self.current_state = new_state

        # Record assistant turn
        self.conversation_history.append(ConversationTurn(
            role="assistant",
            content=response_text,
        ))

        generation_time = (time.time() - start_time) * 1000

        return Response(
            text=response_text,
            state_updated=True,
            skills_active=list(self.loaded_skills.keys()),
            generation_time_ms=generation_time,
        )

    def learn(self, content: str, prefix: str = "Information") -> bool:
        """
        Learn new information into state (in-session learning).

        Unlike skills (which are saved), this learning only persists
        for the current session.

        Args:
            content: Text to learn
            prefix: Prefix for the learning prompt

        Returns:
            True if learning successful
        """
        prompt = f"{prefix}: {content}"
        self.current_state = self.kernel.learn_stream(prompt, self.current_state)
        return True

    def learn_and_save_skill(
        self,
        content: str,
        skill_name: str,
        description: str = ""
    ) -> str:
        """
        Learn content and save as a reusable skill.

        Args:
            content: Text to learn
            skill_name: Name for the skill
            description: Optional description

        Returns:
            Path to saved skill file
        """
        # Save current state as base
        base_state = self.current_state

        # Learn content
        final_state = self.kernel.learn_stream(content, self.current_state)

        # Compute delta
        delta = self.kernel.compute_delta(final_state, base_state)

        # Save skill
        skill_path = self.skills_dir / f"{skill_name}.psa"
        self.kernel.save_state(delta, str(skill_path))

        # Save metadata
        meta_path = self.skills_dir / f"{skill_name}.json"
        metadata = {
            "name": skill_name,
            "description": description,
            "created_at": time.time(),
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
        }
        meta_path.write_text(json.dumps(metadata, indent=2))

        # Update current state
        self.current_state = final_state

        return str(skill_path)

    def reset_conversation(self):
        """
        Reset conversation but keep loaded skills.

        Useful for starting a new topic while retaining skill knowledge.
        """
        self.conversation_history = []

        # Rebuild state from skills only
        self.current_state = None
        for skill_name, weight in self.loaded_skills.items():
            skill_path = self.skills_dir / f"{skill_name}.psa"
            if skill_path.exists():
                delta = self.kernel.load_state(str(skill_path))
                self.current_state = self.kernel.apply_delta(
                    self.current_state, delta, weight=weight
                )

        # Re-apply system prompt if set
        if self.system_prompt:
            self.current_state = self.kernel.learn_stream(
                f"System: {self.system_prompt}",
                self.current_state
            )

    def reset_all(self):
        """
        Full reset - clear state, skills, and conversation.
        """
        self.current_state = None
        self.base_state = None
        self.loaded_skills = {}
        self.conversation_history = []
        self.system_prompt = None

    def save_session(self, filepath: str):
        """
        Save current session state to file.

        Saves:
        - Current state
        - Loaded skills info
        - Conversation history
        """
        session_data = {
            "loaded_skills": self.loaded_skills,
            "system_prompt": self.system_prompt,
            "conversation_history": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp,
                }
                for turn in self.conversation_history
            ],
            "settings": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

        # Save state tensor
        if self.current_state is not None:
            state_path = filepath + ".state"
            self.kernel.save_state(self.current_state, state_path)
            session_data["state_path"] = state_path

        # Save session metadata
        Path(filepath).write_text(json.dumps(session_data, indent=2))

    def load_session(self, filepath: str):
        """
        Load a saved session.
        """
        session_data = json.loads(Path(filepath).read_text())

        # Restore settings
        settings = session_data.get("settings", {})
        self.temperature = settings.get("temperature", 0.8)
        self.max_tokens = settings.get("max_tokens", 256)
        self.top_p = settings.get("top_p", 0.85)

        # Restore skills info
        self.loaded_skills = session_data.get("loaded_skills", {})
        self.system_prompt = session_data.get("system_prompt")

        # Restore conversation history
        self.conversation_history = [
            ConversationTurn(
                role=turn["role"],
                content=turn["content"],
                timestamp=turn.get("timestamp", time.time()),
            )
            for turn in session_data.get("conversation_history", [])
        ]

        # Restore state
        if "state_path" in session_data:
            self.current_state = self.kernel.load_state(session_data["state_path"])

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        status = {
            "has_state": self.current_state is not None,
            "loaded_skills": self.loaded_skills,
            "skill_count": len(self.loaded_skills),
            "conversation_turns": len(self.conversation_history),
            "system_prompt_set": self.system_prompt is not None,
            "settings": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

        if self.current_state is not None:
            state_stats = self.kernel.get_state_stats(self.current_state)
            status["state_size_mb"] = state_stats["size_mb"]

        return status

    def list_available_skills(self) -> List[Dict[str, Any]]:
        """List all available skills in skills directory."""
        skills = []

        for skill_path in self.skills_dir.glob("*.psa"):
            skill_name = skill_path.stem
            meta_path = self.skills_dir / f"{skill_name}.json"

            skill_info = {
                "name": skill_name,
                "path": str(skill_path),
                "size_mb": skill_path.stat().st_size / (1024 * 1024),
                "loaded": skill_name in self.loaded_skills,
            }

            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                skill_info["description"] = meta.get("description", "")
                skill_info["created_at"] = meta.get("created_at")

            skills.append(skill_info)

        return skills
