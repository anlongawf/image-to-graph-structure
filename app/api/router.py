from fastapi import APIRouter
from .endpoints import extract, auth, history, public

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(extract.router, prefix="/extract", tags=["Extract"])
router.include_router(history.router, prefix="/history", tags=["History"])
router.include_router(public.router, prefix="/public", tags=["Public API"])
