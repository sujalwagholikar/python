"""
agents package
================
Specialised, single-app autonomous agents for JARVIS, plus the
ToolSelectorAgent that automatically routes an utterance to the correct
one, RecoveryAgent that restarts a crashed target app and resumes the
task, and LearningAgent that persists lightweight usage patterns across
sessions.

    from agents import ToolSelectorAgent, WhatsAppAgent, GmailAgent, FilesAgent
    from agents import RecoveryAgent, LearningAgent

Each specialist agent is exclusively responsible for its own app -- e.g.
only WhatsAppAgent ever touches WhatsApp. ToolSelectorAgent enforces this
at the routing layer; SpecialistAgent (base_agent.py) enforces it again at
runtime via an app-scope guard. RecoveryAgent and LearningAgent are cross-
cutting: every route() call is automatically wrapped in crash recovery and
(if a LearningAgent is supplied) usage-pattern recording -- no extra calls
needed at the call site.
"""

from agents.base_agent import AgentOp, AgentResult, SpecialistAgent, SpecialistAgentError
from agents.whatsapp_agent import WhatsAppAgent
from agents.gmail_agent import GmailAgent
from agents.files_agent import FilesAgent
from agents.recovery_agent import RecoveryAgent
from agents.learning_agent import LearningAgent
from agents.tool_selector import ToolSelectorAgent

__all__ = [
    "AgentOp",
    "AgentResult",
    "SpecialistAgent",
    "SpecialistAgentError",
    "WhatsAppAgent",
    "GmailAgent",
    "FilesAgent",
    "RecoveryAgent",
    "LearningAgent",
    "ToolSelectorAgent",
]