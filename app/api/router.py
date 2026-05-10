from fastapi import APIRouter

from app.api.endpoints import health, translate

router = APIRouter()
router.include_router(translate.router)
router.include_router(health.router)
