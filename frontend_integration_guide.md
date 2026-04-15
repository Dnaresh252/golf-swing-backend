# Golf Swing AI Platform — Frontend Integration Guide

> **Base URL:** `https://golf-swing-backend-production.up.railway.app`  
> **API Prefix:** `/api/v1`  
> **Interactive Docs:** `https://golf-swing-backend-production.up.railway.app/docs`  
> **Test Account:** `frontend_dev@golftest.com` / `Frontend@1234`

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Key Endpoints by Screen](#2-key-endpoints-by-screen)
3. [File Upload Guide](#3-file-upload-guide)
4. [Submission Status Flow](#4-submission-status-flow)
5. [Error Handling](#5-error-handling)

---

## 1. Getting Started

### Standard Response Envelope

Every API response — success or error — uses this consistent wrapper:

```json
{
  "status": "success",
  "message": "Human-readable message",
  "data": { ... }
}
```

On error:
```json
{
  "status": "error",
  "message": "What went wrong",
  "request_id": "uuid-for-support"
}
```

> Always read `data` for payload. Read `message` for user-facing text. Read `request_id` when reporting bugs.

---

### Authentication Flow

**Step 1 — Register or Login**

```http
POST /api/v1/auth/register
POST /api/v1/auth/login
```

Both return two tokens:

```json
{
  "status": "success",
  "data": {
    "tokens": {
      "access_token": "eyJhbGci...",
      "refresh_token": "eyJhbGci...",
      "token_type": "bearer",
      "expires_in": 1800
    },
    "user": {
      "id": "e8fb04d6-...",
      "email": "user@example.com",
      "name": "Jane Smith",
      "is_verified": false
    }
  }
}
```

| Token | Lifetime | Purpose |
|---|---|---|
| `access_token` | 30 minutes | All API requests |
| `refresh_token` | 7 days | Get new access token |

**Step 2 — Attach token to every request**

```javascript
// axios interceptor (recommended)
axios.interceptors.request.use(config => {
  const token = localStorage.getItem('access_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});
```

**Step 3 — Store tokens**

```javascript
// After login/register
const { tokens, user } = response.data.data;
localStorage.setItem('access_token', tokens.access_token);
localStorage.setItem('refresh_token', tokens.refresh_token);
localStorage.setItem('user', JSON.stringify(user));

// Token expiry tracking
const expiresAt = Date.now() + (tokens.expires_in * 1000);
localStorage.setItem('token_expires_at', expiresAt);
```

---

### Token Refresh Logic

Refresh **before** the access token expires — do not wait for a 401.

```javascript
// utils/auth.js
async function getValidToken() {
  const expiresAt = parseInt(localStorage.getItem('token_expires_at'));
  const fiveMinutes = 5 * 60 * 1000;

  if (Date.now() > expiresAt - fiveMinutes) {
    await refreshToken();
  }

  return localStorage.getItem('access_token');
}

async function refreshToken() {
  const refreshToken = localStorage.getItem('refresh_token');
  if (!refreshToken) {
    redirectToLogin();
    return;
  }

  try {
    const res = await axios.post('/api/v1/auth/refresh', {
      refresh_token: refreshToken,
    });
    const { access_token, expires_in } = res.data.data;
    localStorage.setItem('access_token', access_token);
    localStorage.setItem('token_expires_at', Date.now() + expires_in * 1000);
  } catch {
    // Refresh token expired — force re-login
    localStorage.clear();
    redirectToLogin();
  }
}
```

**Axios response interceptor for automatic refresh:**

```javascript
axios.interceptors.response.use(
  response => response,
  async error => {
    const original = error.config;
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true;
      await refreshToken();
      original.headers.Authorization = `Bearer ${localStorage.getItem('access_token')}`;
      return axios(original);
    }
    return Promise.reject(error);
  }
);
```

---

## 2. Key Endpoints by Screen

### Login / Register Screen

```javascript
// Register
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "password": "SecurePass@123",   // min 8 chars, must include uppercase + number + symbol
  "name": "Jane Smith",
  "terms_accepted": true           // required — must be true
}
// → 201 Created

// Login
POST /api/v1/auth/login
{ "email": "user@example.com", "password": "SecurePass@123" }
// → 200 OK

// Coach Login (separate endpoint — returns coach profile too)
POST /api/v1/auth/coach-login
{ "email": "coach@example.com", "password": "..." }
// → 200 OK — data.coach contains coach profile

// Logout
POST /api/v1/auth/logout
// Authorization: Bearer <token>   (no body needed)
// → 200 OK

// Refresh
POST /api/v1/auth/refresh
{ "refresh_token": "eyJhbGci..." }
// → 200 OK — data.access_token
```

**Rate limits:** Login is capped at **5 requests/minute** per IP. Register is capped at **10 requests/hour**. Exceeding these returns `429 Too Many Requests`.

---

### Dashboard Screen

```javascript
// Get current user profile + stats
GET /api/v1/users/profile
// → data.stats: { total_submissions, completed_submissions, active_discount_codes }

// List submissions (newest first)
GET /api/v1/submissions?page=1&limit=20
// → data: { items: [...], total, page, pages, limit }

// Each submission item:
// { id, status, created_at, updated_at, files: [...] }
```

**Submission status values** (display these to the user):

| Value | Display text |
|---|---|
| `PENDING` | Getting ready... |
| `IMAGES_UPLOADED` | Images received |
| `VIDEO_UPLOADED` | Video received |
| `SUBMITTED` | Submitted for analysis |
| `PROCESSING` | AI is analyzing your swing... |
| `AVATAR_GENERATED` | Avatar ready — awaiting coach review |
| `READY_FOR_REVIEW` | Waiting for coach |
| `IN_REVIEW` | Coach is reviewing |
| `CORRECTIONS_MADE` | Coach corrections available |
| `COMPLETED` | Analysis complete |
| `REJECTED` | Needs resubmission |

---

### File Upload Screen

> Full upload flow is documented in **Section 3**. Here is the API sequence:

```javascript
// 1. Create empty submission (do this first)
POST /api/v1/submissions/create
// → data.id  ← save this as submissionId

// 2. Upload 4 images (all at once, multipart/form-data)
POST /api/v1/submissions/{submissionId}/upload-images

// 3. Upload swing video
POST /api/v1/submissions/{submissionId}/upload-video

// 4. Submit for analysis (triggers AI pipeline)
POST /api/v1/submissions/{submissionId}/submit-for-analysis
```

---

### Avatar Viewer Screen

Available **after** status reaches `AVATAR_GENERATED` or later.

```javascript
// Get avatar (check status field in response)
GET /api/v1/avatar/{submissionId}/avatar
// 202 = still processing, 200 = ready
// data: { id, status, glb_url, obj_url, fbx_url, angle_views: {...} }

// Get a specific angle image
GET /api/v1/avatar/{submissionId}/avatar/angles/front
// angles: top | front | left | right | back
// data: { angle, image_url }

// Get raw skeleton JSON (for 3D viewers / Unity)
GET /api/v1/avatar/{submissionId}/avatar/skeleton
// data: { submission_id, frames: [{ frame_num, timestamp, joints: [...] }] }
```

**Polling the avatar while processing:**

```javascript
async function pollAvatar(submissionId, onReady) {
  const interval = setInterval(async () => {
    const res = await api.get(`/avatar/${submissionId}/avatar`);
    if (res.status === 200 && res.data.data.status === 'COMPLETED') {
      clearInterval(interval);
      onReady(res.data.data);
    }
  }, 5000); // Poll every 5 seconds
}
```

---

### Corrections / Results Screen

Available **after** status reaches `CORRECTIONS_MADE`.

```javascript
// Get all 5 corrected angle videos + coach notes
GET /api/v1/corrections/{submissionId}/corrections
// data: {
//   angles: { front: { video_url }, left: { video_url }, ... },
//   coach_notes: { notes_text, recommendations }
// }

// Get a single angle
GET /api/v1/corrections/{submissionId}/corrections/front
// angles: top | front | left | right | back
```

---

### Coach Screens

> Requires a **coach account** — regular users receive `403 Coach account required`.  
> Use `POST /api/v1/auth/coach-login` to authenticate.

```javascript
// Review queue
GET /api/v1/coach/queue?page=1&limit=20
GET /api/v1/coach/queue?status_filter=READY_FOR_REVIEW

// Open a submission (acquires review lock)
GET /api/v1/coach/queue/{submissionId}
// Returns full submission data: files, avatar, skeleton, existing notes

// Auto-save notes while reviewing
POST /api/v1/coach/queue/{submissionId}/notes
{
  "notes_text": "Good hip rotation. Work on follow-through.",
  "recommendations": ["Keep left arm straight", "Rotate hips earlier"],
  "corrected_skeleton_json": { ... }   // optional — from Unity tool
}

// Upload corrected angle video
POST /api/v1/coach/queue/{submissionId}/upload-corrected-video?angle=front
// Body: multipart/form-data with field: file

// Approve submission
POST /api/v1/coach/queue/{submissionId}/approve

// Reject with reason
POST /api/v1/coach/queue/{submissionId}/reject
{ "reason": "Video quality too low to analyse." }

// Social media approval queue
GET /api/v1/coach/results-queue
POST /api/v1/coach/results-queue/{submissionId}/approve-for-posting
POST /api/v1/coach/results-queue/{submissionId}/reject-posting
{ "reason": "Content not suitable." }
```

---

### Practice Mode Screen

Available **after** coach has saved corrections (`CORRECTIONS_MADE` or later).

```javascript
// Get target skeleton + key frames for practice overlay
GET /api/v1/practice/{submissionId}/practice-mode
// data: {
//   corrected_skeleton_json: { joints: [...] },
//   original_skeleton_json:  { joints: [...] },
//   key_frames: {
//     address:        { phase, joints: [...] },
//     top:            { phase, joints: [...] },
//     impact:         { phase, joints: [...] },
//     follow_through: { phase, joints: [...] }
//   },
//   accuracy_zones: [
//     { joint_id, joint_name, acceptable_range, target_x, target_y, target_z }
//   ]
// }

// Save a practice recording
POST /api/v1/practice/{submissionId}/practice-mode/record
// Body: multipart/form-data, field: file (MP4/MOV, max 100MB)

// List past practice sessions
GET /api/v1/practice/{submissionId}/practice-sessions?page=1&limit=20
```

---

### Social Sharing Screen

```javascript
// Opt in to social sharing
POST /api/v1/social/{submissionId}/social-opt-in
{
  "platforms": ["youtube", "tiktok"],   // one or both
  "face_privacy": "show"                // "show" | "blur" | "hide"
}

// Upload the final results video (mp4/mov, max 100 MB)
POST /api/v1/social/{submissionId}/post-results
// Body: multipart/form-data, field: file

// Trigger posting after coach approval
POST /api/v1/social/{submissionId}/post-to-social
// → 202 Accepted (queued for background posting)

// OAuth — connect YouTube account
GET /api/v1/auth/youtube/url          // requires auth — returns { url }
GET /api/v1/auth/youtube/callback     // ?code=...&state=...  (callback from Google)

// OAuth — connect TikTok account
GET /api/v1/auth/tiktok/url           // requires auth — returns { url }
GET /api/v1/auth/tiktok/callback      // ?code=...&state=...  (callback from TikTok)

// Discount codes (earned after coach approves social post)
GET /api/v1/discounts/discount-codes
// data: { items: [...], total, active_count, used_count }

// Generate discount for a completed submission
POST /api/v1/discounts/{submissionId}/generate-discount
// → 201 Created — data: { code, discount_percent, valid_until }
```

---

## 3. File Upload Guide

### Overview

| Upload | Endpoint | Field name(s) | Format | Max size |
|---|---|---|---|---|
| 4 swing images | `POST /submissions/{id}/upload-images` | `front_image`, `left_image`, `right_image`, `back_image` | JPEG, PNG | 10 MB each |
| Swing video | `POST /submissions/{id}/upload-video` | `swing_video` | MP4, MOV | 100 MB / 15 s |
| Practice recording | `POST /practice/{id}/practice-mode/record` | `file` | MP4, MOV | 100 MB |
| Results video | `POST /social/{id}/post-results` | `file` | MP4, MOV | 100 MB |

---

### Uploading 4 Images

All 4 angles must be sent **in a single request** as `multipart/form-data`.

```javascript
async function uploadImages(submissionId, files) {
  // files = { front, left, right, back }  — each is a File object
  const formData = new FormData();
  formData.append('front_image', files.front);
  formData.append('left_image',  files.left);
  formData.append('right_image', files.right);
  formData.append('back_image',  files.back);

  const res = await axios.post(
    `/api/v1/submissions/${submissionId}/upload-images`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: e => {
        const percent = Math.round((e.loaded / e.total) * 100);
        setProgress(percent); // update your progress bar
      },
    }
  );

  return res.data.data; // updated submission object
}
```

**Client-side validation before uploading:**

```javascript
function validateImage(file) {
  const allowed = ['image/jpeg', 'image/png'];
  const maxBytes = 10 * 1024 * 1024; // 10 MB

  if (!allowed.includes(file.type)) {
    throw new Error('Image must be JPEG or PNG.');
  }
  if (file.size > maxBytes) {
    throw new Error('Image must be under 10 MB.');
  }
}
```

---

### Uploading a Video

```javascript
async function uploadVideo(submissionId, file, onProgress) {
  const formData = new FormData();
  formData.append('swing_video', file);

  const res = await axios.post(
    `/api/v1/submissions/${submissionId}/upload-video`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: e => {
        const percent = Math.round((e.loaded / e.total) * 100);
        onProgress(percent);
      },
      timeout: 120000, // 2 min timeout for large videos
    }
  );

  return res.data.data;
}
```

**Client-side video validation:**

```javascript
function validateVideo(file) {
  const allowed = ['video/mp4', 'video/quicktime'];
  const maxBytes = 100 * 1024 * 1024; // 100 MB

  if (!allowed.includes(file.type)) {
    throw new Error('Video must be MP4 or MOV.');
  }
  if (file.size > maxBytes) {
    throw new Error('Video must be under 100 MB.');
  }
}

// Optional: check duration before upload (requires browser API)
function checkVideoDuration(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src);
      if (video.duration > 15) {
        reject(new Error('Video must be 15 seconds or less.'));
      } else {
        resolve(video.duration);
      }
    };
    video.src = URL.createObjectURL(file);
  });
}
```

---

### Complete Upload Flow (React example)

```javascript
async function submitSwing({ frontImage, leftImage, rightImage, backImage, video }) {

  // Step 1 — Create submission
  const createRes = await api.post('/submissions/create');
  const submissionId = createRes.data.data.id;

  // Step 2 — Upload images
  setStatus('Uploading images...');
  await uploadImages(submissionId, {
    front: frontImage,
    left:  leftImage,
    right: rightImage,
    back:  backImage,
  });

  // Step 3 — Upload video
  setStatus('Uploading video...');
  await uploadVideo(submissionId, video, pct => setVideoProgress(pct));

  // Step 4 — Submit for AI analysis
  setStatus('Submitting for analysis...');
  await api.post(`/submissions/${submissionId}/submit-for-analysis`);

  // Step 5 — Poll for status updates (see Section 4)
  startPolling(submissionId);
}
```

---

## 4. Submission Status Flow

### Status Lifecycle

```
PENDING
  │
  ├─ upload-images → IMAGES_UPLOADED
  │
  ├─ upload-video  → VIDEO_UPLOADED
  │
  ├─ submit-for-analysis → SUBMITTED
  │                             │
  │                        [AI Pipeline running — ~2 min]
  │                             │
  │                        PROCESSING
  │                             │
  │                        AVATAR_GENERATED
  │                             │
  │                        READY_FOR_REVIEW
  │                             │
  │                        IN_REVIEW (coach opens submission)
  │                             │
  │                    ┌────────┴────────┐
  │               CORRECTIONS_MADE   REJECTED
  │                    │
  │               COMPLETED
  │
  └─ DELETE (only allowed in PENDING or REJECTED)
```

### Polling for Status Updates

The API uses **HTTP polling** (no WebSocket). Poll the status endpoint on an interval:

```javascript
// hooks/useSubmissionStatus.js
import { useState, useEffect, useRef } from 'react';
import api from '../utils/api';

const TERMINAL_STATUSES = new Set([
  'CORRECTIONS_MADE', 'COMPLETED', 'REJECTED'
]);

const POLL_INTERVAL_MS = 5000; // 5 seconds

export function useSubmissionStatus(submissionId) {
  const [status, setStatus]   = useState(null);
  const [message, setMessage] = useState('');
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!submissionId) return;

    const poll = async () => {
      try {
        const res = await api.get(`/submissions/${submissionId}/status`);
        const { status: s, message: m } = res.data.data;
        setStatus(s);
        setMessage(m);

        if (TERMINAL_STATUSES.has(s)) {
          clearInterval(intervalRef.current);
        }
      } catch (err) {
        console.error('Status poll failed:', err);
      }
    };

    poll(); // immediate first check
    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => clearInterval(intervalRef.current);
  }, [submissionId]);

  return { status, message };
}
```

**Usage in a component:**

```jsx
function AnalysisProgress({ submissionId }) {
  const { status, message } = useSubmissionStatus(submissionId);

  const progressMap = {
    PENDING:            0,
    IMAGES_UPLOADED:   20,
    VIDEO_UPLOADED:    35,
    SUBMITTED:         40,
    PROCESSING:        55,
    AVATAR_GENERATED:  70,
    READY_FOR_REVIEW:  75,
    IN_REVIEW:         80,
    CORRECTIONS_MADE:  95,
    COMPLETED:        100,
    REJECTED:           0,
  };

  return (
    <div>
      <p>{message}</p>
      <ProgressBar value={progressMap[status] ?? 0} />
      {status === 'CORRECTIONS_MADE' && (
        <button onClick={() => navigate(`/corrections/${submissionId}`)}>
          View Coach Corrections
        </button>
      )}
      {status === 'REJECTED' && (
        <p className="error">Rejected — please resubmit with better footage.</p>
      )}
    </div>
  );
}
```

### What to Show on Each Status

| Status | Show the user |
|---|---|
| `PENDING` | "Preparing your submission" |
| `IMAGES_UPLOADED` | "Images uploaded — waiting for video" |
| `VIDEO_UPLOADED` | "Video ready — tap Submit when ready" |
| `SUBMITTED` | "Submitted! AI analysis starting..." |
| `PROCESSING` | Spinner + "AI is analysing your swing" |
| `AVATAR_GENERATED` | "Avatar built — in coach queue" |
| `READY_FOR_REVIEW` | "Waiting for your coach" |
| `IN_REVIEW` | "Coach is reviewing now" |
| `CORRECTIONS_MADE` | ✅ "Your coach has corrections ready!" + CTA button |
| `COMPLETED` | ✅ "Analysis complete!" + Share / Download CTAs |
| `REJECTED` | ❌ "Needs resubmission" + reason from coach notes |

---

## 5. Error Handling

### Standard Error Format

```json
{
  "status": "error",
  "message": "Friendly description for the user",
  "request_id": "7b153356-d514-4487-bf56-b0ad1faf60a7"
}
```

Validation errors (`422`) include a field breakdown:

```json
{
  "status": "error",
  "message": "Validation failed. Please check your input.",
  "errors": [
    { "field": "body -> email",    "message": "value is not a valid email address", "type": "value_error" },
    { "field": "body -> password", "message": "min 8 characters required",          "type": "string_too_short" }
  ],
  "request_id": "..."
}
```

---

### Common HTTP Status Codes

| Code | Meaning | What to do |
|---|---|---|
| `200` | Success | Read `data` |
| `201` | Created | Resource created — read `data.id` |
| `202` | Accepted | Background task queued — start polling |
| `400` | Bad request | Show `message` to user |
| `401` | Unauthorised | Token expired/missing — refresh or redirect to login |
| `403` | Forbidden | Wrong account type (e.g. non-coach on coach endpoint) |
| `404` | Not found | Submission/resource doesn't exist |
| `409` | Conflict | Email already registered |
| `422` | Validation error | Show `errors` array inline on form fields |
| `423` | Locked | Account locked after 5 failed login attempts — show 30-min lockout message |
| `429` | Rate limit | Show "Too many attempts — please wait" with a countdown |
| `500` | Server error | Show generic error + log `request_id` |

---

### Global Error Handler (axios)

```javascript
// utils/api.js
import axios from 'axios';

const api = axios.create({
  baseURL: 'https://golf-swing-backend-production.up.railway.app/api/v1',
});

// Attach token
api.interceptors.request.use(config => {
  const token = localStorage.getItem('access_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Handle errors globally
api.interceptors.response.use(
  response => response,
  async error => {
    const { response, config } = error;

    if (!response) {
      // Network error
      showToast('Network error — please check your connection.');
      return Promise.reject(error);
    }

    const { status, data } = response;
    const message = data?.message || 'Something went wrong.';
    const requestId = data?.request_id;

    switch (status) {
      case 401:
        // Try refresh first
        if (!config._retry) {
          config._retry = true;
          try {
            await refreshToken();
            config.headers.Authorization = `Bearer ${localStorage.getItem('access_token')}`;
            return api(config);
          } catch {
            localStorage.clear();
            window.location.href = '/login';
          }
        }
        break;

      case 403:
        showToast('You do not have permission to do this.');
        break;

      case 409:
        showToast('This email address is already registered.');
        break;

      case 422:
        // Pass validation errors back to the form
        return Promise.reject({ validationErrors: data?.errors, message });

      case 423:
        showToast('Account locked after too many failed attempts. Try again in 30 minutes.');
        break;

      case 429:
        showToast('Too many requests — please wait a moment and try again.');
        break;

      case 500:
        showToast(`Server error. Reference: ${requestId ?? 'unknown'}`);
        console.error('Server error request_id:', requestId);
        break;

      default:
        showToast(message);
    }

    return Promise.reject(error);
  }
);

export default api;
```

---

### Handling 422 Validation Errors in Forms

```jsx
// React Hook Form example
async function onSubmit(data) {
  try {
    await api.post('/auth/register', data);
    navigate('/dashboard');
  } catch (err) {
    if (err.validationErrors) {
      // Map API field errors back to form fields
      err.validationErrors.forEach(({ field, message }) => {
        // field format: "body -> email"
        const fieldName = field.split(' -> ').pop();
        setError(fieldName, { type: 'server', message });
      });
    } else {
      showToast(err.response?.data?.message || 'Registration failed.');
    }
  }
}
```

---

## Quick Reference

```
Auth
  POST   /api/v1/auth/register
  POST   /api/v1/auth/login
  POST   /api/v1/auth/coach-login
  POST   /api/v1/auth/refresh
  POST   /api/v1/auth/logout

User
  GET    /api/v1/users/profile
  PUT    /api/v1/users/profile
  PUT    /api/v1/users/settings

Submissions
  POST   /api/v1/submissions/create
  GET    /api/v1/submissions
  GET    /api/v1/submissions/{id}
  GET    /api/v1/submissions/{id}/status         ← poll this
  POST   /api/v1/submissions/{id}/upload-images
  POST   /api/v1/submissions/{id}/upload-video
  POST   /api/v1/submissions/{id}/submit-for-analysis
  DELETE /api/v1/submissions/{id}

Avatar
  GET    /api/v1/avatar/{id}/avatar
  GET    /api/v1/avatar/{id}/avatar/angles/{angle}
  GET    /api/v1/avatar/{id}/avatar/skeleton

Corrections
  GET    /api/v1/corrections/{id}/corrections
  GET    /api/v1/corrections/{id}/corrections/{angle}

Practice
  GET    /api/v1/practice/{id}/practice-mode
  POST   /api/v1/practice/{id}/practice-mode/record
  GET    /api/v1/practice/{id}/practice-sessions

Social
  POST   /api/v1/social/{id}/social-opt-in
  POST   /api/v1/social/{id}/post-results
  POST   /api/v1/social/{id}/post-to-social

Discounts
  GET    /api/v1/discounts/discount-codes
  POST   /api/v1/discounts/{id}/generate-discount

OAuth
  GET    /api/v1/auth/youtube/url
  GET    /api/v1/auth/tiktok/url

Coach (requires coach account)
  GET    /api/v1/coach/queue
  GET    /api/v1/coach/queue/{id}
  POST   /api/v1/coach/queue/{id}/notes
  POST   /api/v1/coach/queue/{id}/approve
  POST   /api/v1/coach/queue/{id}/reject
  POST   /api/v1/coach/queue/{id}/upload-corrected-video
  GET    /api/v1/coach/results-queue
  POST   /api/v1/coach/results-queue/{id}/approve-for-posting
  POST   /api/v1/coach/results-queue/{id}/reject-posting
```

---

*Guide generated from live API — commit `691bd97` · Golf Swing AI Platform v1.0.0*
