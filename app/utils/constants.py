from enum import Enum


# ---------------------------------------------------------------------------
# Status / type enums (mirrors model enums for use outside ORM layer)
# ---------------------------------------------------------------------------

class SubmissionStatus(str, Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    ANALYZING = "ANALYZING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    IN_REVIEW = "IN_REVIEW"
    CORRECTIONS_MADE = "CORRECTIONS_MADE"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"


class FileType(str, Enum):
    FRONT_IMAGE = "FRONT_IMAGE"
    LEFT_IMAGE = "LEFT_IMAGE"
    RIGHT_IMAGE = "RIGHT_IMAGE"
    BACK_IMAGE = "BACK_IMAGE"
    SWING_VIDEO = "SWING_VIDEO"


class AvatarStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CorrectionAngle(str, Enum):
    TOP = "TOP"
    FRONT = "FRONT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BACK = "BACK"


class FacePrivacy(str, Enum):
    SHOW = "SHOW"
    BLUR = "BLUR"
    COVER = "COVER"


# ---------------------------------------------------------------------------
# Email template names
# ---------------------------------------------------------------------------

class EmailTemplate(str, Enum):
    WELCOME = "welcome"
    UPLOAD_CONFIRM = "upload_confirm"
    ANALYSIS_COMPLETE = "analysis_complete"
    COACH_APPROVAL = "coach_approval"
    DISCOUNT_EARNED = "discount_earned"


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

class ErrorMessage:
    # Auth
    INVALID_CREDENTIALS = "Invalid email or password."
    ACCOUNT_LOCKED = "Account temporarily locked. Please try again in 30 minutes."
    ACCOUNT_INACTIVE = "Your account has been deactivated. Contact support."
    ACCOUNT_NOT_VERIFIED = "Please verify your email before logging in."
    TOKEN_EXPIRED = "Your session has expired. Please log in again."
    TOKEN_INVALID = "Invalid authentication token."
    TOKEN_MISSING = "Authentication token is required."
    INSUFFICIENT_PERMISSIONS = "You do not have permission to perform this action."

    # User
    USER_NOT_FOUND = "User not found."
    EMAIL_ALREADY_REGISTERED = "An account with this email already exists."
    PROFILE_UPDATE_FAILED = "Failed to update profile. Please try again."

    # Submission
    SUBMISSION_NOT_FOUND = "Submission not found."
    SUBMISSION_ACCESS_DENIED = "You do not have access to this submission."
    SUBMISSION_INVALID_STATUS = "This action is not allowed in the current submission status."

    # Files
    FILE_TOO_LARGE_IMAGE = "Image exceeds the maximum allowed size of 10 MB."
    FILE_TOO_LARGE_VIDEO = "Video exceeds the maximum allowed size of 100 MB."
    VIDEO_TOO_LONG = "Video exceeds the maximum allowed duration of 15 seconds."
    INVALID_IMAGE_TYPE = "Invalid image format. Allowed: JPEG, PNG."
    INVALID_VIDEO_TYPE = "Invalid video format. Allowed: MP4, MOV, WebM."
    FILE_UPLOAD_FAILED = "File upload failed. Please try again."

    # Coach
    COACH_NOT_FOUND = "Coach profile not found."
    COACH_ALREADY_EXISTS = "A coach profile already exists for this account."
    NOT_A_COACH = "This action requires a coach account."

    # Avatar
    AVATAR_NOT_FOUND = "Avatar not found for this submission."
    AVATAR_PROCESSING = "Avatar is still being processed. Please check back shortly."

    # Discount
    DISCOUNT_NOT_FOUND = "Discount code not found."
    DISCOUNT_ALREADY_USED = "This discount code has already been used."
    DISCOUNT_EXPIRED = "This discount code has expired."
    DISCOUNT_INVALID_FORMAT = "Invalid discount code format."

    # Social
    SOCIAL_SHARE_FAILED = "Failed to share to social media. Please try again."

    # Generic
    INTERNAL_ERROR = "An unexpected error occurred. Please try again later."
    VALIDATION_ERROR = "Invalid input. Please check your data and try again."
    NOT_FOUND = "The requested resource was not found."
    RATE_LIMITED = "Too many requests. Please slow down."


# ---------------------------------------------------------------------------
# Success messages
# ---------------------------------------------------------------------------

class SuccessMessage:
    REGISTRATION_COMPLETE = "Registration successful. Please verify your email."
    LOGIN_SUCCESS = "Logged in successfully."
    LOGOUT_SUCCESS = "Logged out successfully."
    PASSWORD_RESET_SENT = "Password reset instructions sent to your email."
    PASSWORD_UPDATED = "Password updated successfully."
    PROFILE_UPDATED = "Profile updated successfully."
    SUBMISSION_CREATED = "Submission created successfully."
    FILES_UPLOADED = "Files uploaded successfully."
    ANALYSIS_SUBMITTED = "Your swing is being analyzed."
    COACH_NOTES_SAVED = "Notes saved successfully."
    DISCOUNT_CODE_GENERATED = "Discount code generated successfully."
    SOCIAL_SHARED = "Shared to social media successfully."
    COACH_PROFILE_CREATED = "Coach profile created successfully."


# ---------------------------------------------------------------------------
# Pagination defaults
# ---------------------------------------------------------------------------

DEFAULT_PAGE = 1
DEFAULT_LIMIT = 20
MAX_LIMIT = 100

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

DISCOUNT_CODE_PATTERN = r"^GOLF\d{4}-[A-Z0-9]{5}$"
PASSWORD_MIN_LENGTH = 8
REQUEST_ID_HEADER = "X-Request-ID"
