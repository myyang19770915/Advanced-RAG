"""Common dependencies for API routers."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Require the configured admin token on protected endpoints."""
    settings = get_settings()
    # Allow disabling auth in dev for convenience
    if settings.app_env in {"dev", "test"} and not x_admin_token:
        return
    if not x_admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token"
        )
