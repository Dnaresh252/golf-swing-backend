# Golf Swing AI Platform — Backend API

Enterprise-grade REST API for an AI-powered golf swing instruction platform.
Users upload swing videos and photos, receive 3-D avatar pose analysis, expert
coach review with annotated corrections, real-time status updates via WebSocket,
social sharing, and personalised discount rewards.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.110+ (Python 3.9+) |
| Database | PostgreSQL 15+ with SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| Task Queue | Celery 5 with Redis broker |
| File Storage | Backblaze B2 (b2sdk) |
| Authentication | JWT (python-jose) + bcrypt |
| Email | SendGrid Dynamic Templates |
| Pose Detection | MediaPipe / Specialist 3 ML module |
| Social Video | YouTube Data API v3, TikTok Content Posting API |
| Real-time | WebSocket (FastAPI native) |
| Cache / Locks | Redis (redis-py async) |

---

## Prerequisites

- Python 3.9 or later
- PostgreSQL 15+
- Redis 7+
- FFmpeg (must be on `PATH` — used for video rendering)
- A Backblaze B2 account and bucket
- SendGrid account with dynamic templates configured
- (Optional) YouTube OAuth2 credentials
- (Optional) TikTok developer app credentials

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-org/golf_swing_backend.git
cd golf_swing_backend
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in every value — at minimum:

```
SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/golf_swing_db
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
B2_APPLICATION_KEY_ID=...
B2_APPLICATION_KEY=...
B2_BUCKET_NAME=...
B2_BUCKET_ID=...
SENDGRID_API_KEY=...
SENDGRID_FROM_EMAIL=noreply@yourdomain.com
```

### 5. Create the database and run migrations

```bash
# Create the PostgreSQL database first
createdb golf_swing_db

# Run all Alembic migrations
alembic upgrade head
```

### 6. Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 7. Start the Celery worker

```bash
celery -A celery_worker worker --loglevel=info --concurrency=4
```

### 8. Start the Celery beat scheduler (periodic tasks)

```bash
celery -A celery_worker beat --loglevel=info
```

> For development you can run worker + beat together:
> `celery -A celery_worker worker --beat --loglevel=info`

### 9. Open the interactive API docs

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | Yes | App-level secret (CSRF, misc signing) |
| `DATABASE_URL` | Yes | PostgreSQL async connection string |
| `REDIS_URL` | Yes | Redis connection URL |
| `JWT_SECRET_KEY` | Yes | JWT signing secret |
| `JWT_ALGORITHM` | No | Default: `HS256` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No | Default: `30` |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | No | Default: `7` |
| `B2_APPLICATION_KEY_ID` | Yes | Backblaze B2 key ID |
| `B2_APPLICATION_KEY` | Yes | Backblaze B2 secret key |
| `B2_BUCKET_NAME` | Yes | B2 bucket name |
| `B2_BUCKET_ID` | Yes | B2 bucket ID |
| `MAX_IMAGE_SIZE_MB` | No | Default: `10` |
| `MAX_VIDEO_SIZE_MB` | No | Default: `100` |
| `MAX_VIDEO_DURATION_SECONDS` | No | Default: `15` |
| `SENDGRID_API_KEY` | Yes | SendGrid API key |
| `SENDGRID_FROM_EMAIL` | Yes | Sender email address |
| `SENDGRID_WELCOME_TEMPLATE` | Yes | SendGrid template ID |
| `SENDGRID_UPLOAD_CONFIRM_TEMPLATE` | Yes | SendGrid template ID |
| `SENDGRID_ANALYSIS_COMPLETE_TEMPLATE` | Yes | SendGrid template ID |
| `SENDGRID_COACH_APPROVAL_TEMPLATE` | Yes | SendGrid template ID |
| `SENDGRID_DISCOUNT_EARNED_TEMPLATE` | Yes | SendGrid template ID |
| `YOUTUBE_CLIENT_ID` | No | YouTube OAuth2 client ID |
| `YOUTUBE_CLIENT_SECRET` | No | YouTube OAuth2 secret |
| `TIKTOK_CLIENT_KEY` | No | TikTok app client key |
| `TIKTOK_CLIENT_SECRET` | No | TikTok app secret |
| `DISCOUNT_CODE_PREFIX` | No | Default: `GOLF2024` |
| `DISCOUNT_PERCENTAGE` | No | Default: `15` |
| `DISCOUNT_VALIDITY_DAYS` | No | Default: `30` |
| `FFMPEG_PATH` | No | Default: `ffmpeg` |
| `LOG_LEVEL` | No | Default: `INFO` |
| `LOG_FILE` | No | Default: `logs/app.log` |

See `.env.example` for the full list.

---

## API Endpoints Overview

### Authentication — `/api/v1/auth`

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | User login |
| POST | `/auth/coach-login` | Coach login |
| POST | `/auth/refresh` | Refresh access token |
| POST | `/auth/logout` | Invalidate session |

### Submissions — `/api/v1/submissions`

| Method | Path | Description |
|---|---|---|
| POST | `/submissions/create` | Create empty submission |
| POST | `/submissions/{id}/upload-images` | Upload 4 swing images |
| POST | `/submissions/{id}/upload-video` | Upload swing video |
| POST | `/submissions/{id}/submit-for-analysis` | Trigger AI analysis |
| GET | `/submissions` | List user's submissions |
| GET | `/submissions/{id}` | Get submission detail |
| GET | `/submissions/{id}/status` | Poll status (for frontend) |
| DELETE | `/submissions/{id}` | Delete submission |

### Avatar — `/api/v1/avatar` (proxied as `/submissions/{id}/avatar`)

| Method | Path | Description |
|---|---|---|
| GET | `/submissions/{id}/avatar` | Get avatar data |
| GET | `/submissions/{id}/avatar/skeleton` | Raw skeleton JSON (Unity) |
| GET | `/submissions/{id}/avatar/angles/{angle}` | Single angle view image |

### Corrections — `/api/v1/corrections`

| Method | Path | Description |
|---|---|---|
| GET | `/submissions/{id}/corrections` | All corrected videos + notes |
| GET | `/submissions/{id}/corrections/{angle}` | Single angle correction |

### Coach — `/api/v1/coach`

| Method | Path | Description |
|---|---|---|
| GET | `/coach/queue` | Review queue (paginated) |
| GET | `/coach/queue/{id}` | Open submission (acquires lock) |
| POST | `/coach/queue/{id}/notes` | Save / auto-save notes |
| POST | `/coach/queue/{id}/upload-corrected-video` | Upload corrected video |
| POST | `/coach/queue/{id}/approve` | Approve submission |
| POST | `/coach/queue/{id}/reject` | Reject submission |
| GET | `/coach/results-queue` | Social approval queue |
| POST | `/coach/results-queue/{id}/approve-for-posting` | Approve social post |
| POST | `/coach/results-queue/{id}/reject-posting` | Reject social post |

### Social & Results — `/api/v1/social`

| Method | Path | Description |
|---|---|---|
| POST | `/submissions/{id}/social-opt-in` | Consent to social posting |
| POST | `/submissions/{id}/post-results` | Upload results video |
| POST | `/submissions/{id}/post-to-social` | Trigger social posting |

### Discounts — `/api/v1/discounts`

| Method | Path | Description |
|---|---|---|
| GET | `/user/discount-codes` | List user discount codes |
| POST | `/submissions/{id}/generate-discount` | Generate discount code |

### Practice Mode — `/api/v1/practice`

| Method | Path | Description |
|---|---|---|
| GET | `/submissions/{id}/practice-mode` | Get comparison data |
| POST | `/submissions/{id}/practice-mode/record` | Upload practice recording |
| GET | `/submissions/{id}/practice-sessions` | List practice recordings |

### User Profile — `/api/v1/users`

| Method | Path | Description |
|---|---|---|
| GET | `/user/profile` | Get own profile + stats |
| GET | `/user/profile/{username}` | Public profile (no auth) |
| PUT | `/user/profile` | Update profile |
| PUT | `/user/settings` | Update privacy settings |

### System

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check (DB ping) |
| GET | `/` | Root info |

---

## Submission Lifecycle

```
PENDING
  └─► UPLOADING       (images uploaded)
        └─► ANALYZING      (AI pipeline running — Celery)
              └─► READY_FOR_REVIEW  (avatar complete)
                    └─► IN_REVIEW        (coach opened submission)
                          ├─► CORRECTIONS_MADE  (coach approved)
                          └─► REJECTED          (coach rejected)
                                └─► COMPLETED   (results video uploaded)
```

---

## Team Integration Notes

### Specialist 1 — Frontend Developer
- Full Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- All endpoints return `{"status": "success"/"error", "data": {...}, "message": "..."}`
- Poll `GET /submissions/{id}/status` for real-time progress updates
- Authentication: `Authorization: Bearer <access_token>` header on all protected routes
- Request IDs returned in `X-Request-ID` response header for debugging

### Specialist 3 — ML / MediaPipe Developer
- Integration point: [app/integrations/mediapipe_client.py](app/integrations/mediapipe_client.py)
- Drop your module as `specialist3_ml` anywhere on `PYTHONPATH`
- Expected interface:
  ```python
  specialist3_ml.detect_pose_from_image(image_path: str, angle: str) -> list[dict]
  specialist3_ml.detect_pose_from_video(video_path: str) -> list[dict]
  specialist3_ml.generate_3d_avatar(skeleton_data: dict) -> dict  # {obj_path, fbx_path, glb_path}
  specialist3_ml.render_avatar_angles(model_path: str) -> dict    # {top, front, left, right, back}
  ```
- Without your module the system runs in **mock mode** — all endpoints work with realistic fake data
- Joint format expected: `[{id, name, x, y, z, confidence}, ...]` (33 MediaPipe BlazePose landmarks)

### Specialist 4 — Unity Coach Tool Developer
- Skeleton data endpoint: `GET /submissions/{id}/avatar/skeleton`
- Returns raw `skeleton_json` — same structure Specialist 3 delivers
- Save corrected skeleton back: `POST /coach/queue/{id}/notes` with `corrected_skeleton_json` in body
- Corrected skeleton format: `{"joints": [{id, name, x, y, z, confidence}, ...]}`
- Coach must be authenticated and hold the Redis lock on the submission

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Generating a New Migration

After changing any model:

```bash
alembic revision --autogenerate -m "describe your change"
alembic upgrade head
```

---

## Project Structure

```
golf_swing_backend/
├── app/
│   ├── api/            # FastAPI route handlers
│   ├── models/         # SQLAlchemy ORM models
│   ├── schemas/        # Pydantic request/response schemas
│   ├── services/       # Business logic layer
│   ├── workers/        # Celery async tasks
│   ├── integrations/   # Third-party API clients
│   └── utils/          # Shared helpers, constants, validators
├── alembic/            # Database migration scripts
├── tests/              # Pytest test suite
├── .env.example        # Environment variable template
├── alembic.ini         # Alembic configuration
├── celery_worker.py    # Celery entry-point
└── requirements.txt    # Python dependencies
```
