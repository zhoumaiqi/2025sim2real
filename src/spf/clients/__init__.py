"""
Clients module for SPF (See, Point, Fly)

This module contains clients for interfacing with model services:
- VLMClient: Unified client supporting multiple VLM providers
- RemoteDepthProClient: HTTP client for remote Depth Pro inference
"""

from .depth_pro_client import RemoteDepthProClient
from .vlm_client import VLMClient

__all__ = [
    "RemoteDepthProClient",
    "VLMClient"
]
