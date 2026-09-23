"""Exact pinned branch agent with console narration disabled for evaluation."""
from branch_agent import Agent as BranchAgent


class Agent(BranchAgent):
    def __init__(self):
        super().__init__(verbose=False)
