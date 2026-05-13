import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.avatar import Avatar, AvatarStatus
from app.models.submission import Submission
from app.schemas.avatar import AvatarAngleResponse, AvatarResponse, SkeletonData

logger = logging.getLogger(__name__)

_VALID_ANGLES = {"top", "front", "left", "right", "back"}

# Maps angle name → Avatar column name
_ANGLE_URL_COLUMNS = {
    "top":   "view_top_url",
    "front": "view_front_url",
    "left":  "view_left_url",
    "right": "view_right_url",
    "back":  "view_back_url",
}


class AvatarService:

    # ------------------------------------------------------------------
    # Ownership guard
    # ------------------------------------------------------------------

    async def _get_owned_submission_id(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Raise 404 if submission doesn't exist or doesn't belong to user."""
        result = await db.execute(
            select(Submission.id, Submission.user_id).where(
                Submission.id == submission_id
            )
        )
        row = result.one_or_none()
        if row is None or row.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found.",
            )

    async def _get_avatar_for_submission(
        self, db: AsyncSession, submission_id: uuid.UUID
    ) -> Optional[Avatar]:
        result = await db.execute(
            select(Avatar).where(Avatar.submission_id == submission_id)
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def get_avatar(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Avatar:
        """Verify ownership and return the Avatar. Raises 404 if not found."""
        await self._get_owned_submission_id(db, submission_id, user_id)

        avatar = await self._get_avatar_for_submission(db, submission_id)
        if avatar is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Avatar not ready yet.",
            )
        return avatar

    async def get_skeleton_data(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """
        Return the raw skeleton_json dict.
        Raises 404 if avatar missing, 400 if avatar not yet COMPLETED.
        """
        avatar = await self.get_avatar(db, submission_id, user_id)

        if avatar.status != AvatarStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Skeleton not ready. Avatar processing is not complete.",
            )
        if avatar.skeleton_json is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Skeleton data not found.",
            )
        return avatar.skeleton_json

    async def get_angle_view(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
        angle: str,
    ) -> Dict[str, str]:
        """
        Return {angle, image_url} for the requested angle.
        Raises 400 for invalid angle, 404 if URL not yet stored.
        """
        angle = angle.lower().strip()
        if angle not in _VALID_ANGLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid angle. Use: top, front, left, right, back",
            )

        avatar = await self.get_avatar(db, submission_id, user_id)
        url: Optional[str] = getattr(avatar, _ANGLE_URL_COLUMNS[angle], None)

        if not url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Angle view '{angle}' not ready yet.",
            )
        return {"angle": angle, "image_url": url}

    async def create_avatar_record(
        self, db: AsyncSession, submission_id: uuid.UUID
    ) -> Avatar:
        """Create the initial PENDING avatar record when a submission is created."""
        existing = await self._get_avatar_for_submission(db, submission_id)
        if existing:
            return existing

        avatar = Avatar(
            submission_id=submission_id,
            status=AvatarStatus.PENDING,
        )
        db.add(avatar)
        await db.flush()
        await db.refresh(avatar)
        logger.info("Avatar record created for submission: %s", submission_id)
        return avatar

    async def update_avatar_status(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        new_status: AvatarStatus,
        error: Optional[str] = None,
    ) -> Optional[Avatar]:
        """Update avatar status and optional error message. Returns None if not found."""
        avatar = await self._get_avatar_for_submission(db, submission_id)
        if avatar is None:
            logger.warning(
                "update_avatar_status: no avatar found for submission %s", submission_id
            )
            return None

        avatar.status = new_status
        if error is not None:
            avatar.error_message = error
        await db.flush()
        await db.refresh(avatar)
        logger.info(
            "Avatar status → %s for submission: %s", new_status.value, submission_id
        )
        return avatar

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    def build_avatar_response(self, avatar: Avatar) -> AvatarResponse:
        """Convert an Avatar ORM object to an AvatarResponse schema."""
        angles = []
        for angle, col in _ANGLE_URL_COLUMNS.items():
            url = getattr(avatar, col, None)
            if url:
                angles.append(AvatarAngleResponse(angle=angle, image_url=url))

        skeleton_data = None
        if avatar.skeleton_json and avatar.status == AvatarStatus.COMPLETED:
            try:
                skeleton_data = SkeletonData(
                    submission_id=avatar.submission_id,
                    # Real pipeline stores frames under "frames"; legacy mock data used "video_frames"
                    frames=avatar.skeleton_json.get("frames")
                           or avatar.skeleton_json.get("video_frames", []),
                )
            except Exception:
                skeleton_data = None

        return AvatarResponse(
            id=avatar.id,
            submission_id=avatar.submission_id,
            status=avatar.status.value,
            skeleton_data=skeleton_data,
            avatar_glb_url=avatar.avatar_glb_url,
            avatar_fbx_url=avatar.avatar_fbx_url,
            angles=angles,
            skin_tone_value=avatar.skin_tone_value,
            body_thickness_value=avatar.body_thickness_value,
            attire_selection=avatar.attire_selection,
            shoe_color_value=avatar.shoe_color_value,
            headgear_enabled=avatar.headgear_enabled,
            error_message=avatar.error_message,
            created_at=avatar.created_at,
        )


avatar_service = AvatarService()
