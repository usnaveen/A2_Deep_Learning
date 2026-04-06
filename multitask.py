"""Root-level re-export so the autograder can do:
    from multitask import MultiTaskPerceptionModel
"""
from models.multitask import MultiTaskPerceptionModel

__all__ = ["MultiTaskPerceptionModel"]
