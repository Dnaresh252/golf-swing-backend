from app.models.user import User
from app.models.coach import Coach
from app.models.submission import Submission
from app.models.submission_file import SubmissionFile
from app.models.avatar import Avatar
from app.models.correction import CorrectedVideo
from app.models.coach_notes import CoachNotes
from app.models.results import ResultsVideo
from app.models.social import SocialSharing
from app.models.discount import DiscountCode

__all__ = [
    "User",
    "Coach",
    "Submission",
    "SubmissionFile",
    "Avatar",
    "CorrectedVideo",
    "CoachNotes",
    "ResultsVideo",
    "SocialSharing",
    "DiscountCode",
]
