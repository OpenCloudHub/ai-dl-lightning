# ==============================================================================
# Serving Module
# ==============================================================================
#
# Fashion MNIST model serving with Ray Serve + FastAPI.
#
# Components:
#   - serve.py: Ray Serve deployment with FastAPI
#   - schemas.py: Pydantic request/response models
#   - config.py: Serving configuration
#   - serve_config.yaml: Deployment configuration
#
# Entry Point:
#   serve run src.serving.serve:app_builder model_uri="models:/model/1"
#
# ==============================================================================
"""Serving module for Fashion MNIST classifier."""
