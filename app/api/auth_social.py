from fastapi import APIRouter, Depends, HTTPException
from app.integrations.youtube import youtube_service
from app.integrations.tiktok import tiktok_service
from app.dependencies import get_current_user

router = APIRouter()


@router.get("/auth/youtube/url")
async def get_youtube_auth_url(
    current_user=Depends(get_current_user)
):
    url = youtube_service.get_auth_url(str(current_user.id))
    return {"status": "success", "data": {"url": url}}


@router.get("/auth/youtube/callback")
async def youtube_callback(code: str, state: str = None):
    try:
        tokens = youtube_service.handle_callback(code, state)
        return {"status": "success", "data": tokens}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/auth/tiktok/url")
async def get_tiktok_auth_url(
    current_user=Depends(get_current_user)
):
    url = tiktok_service.get_auth_url(str(current_user.id))
    return {"status": "success", "data": {"url": url}}


@router.get("/auth/tiktok/callback")
async def tiktok_callback(code: str):
    try:
        tokens = tiktok_service.handle_callback(code)
        return {"status": "success", "data": tokens}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
