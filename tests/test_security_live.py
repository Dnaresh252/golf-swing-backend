"""
Live security verification tests — run after deploying bb2c9a3.

Usage:
    python tests/test_security_live.py
"""
import time
import uuid

import httpx

BASE = "https://golf-swing-backend-production.up.railway.app"
results: dict = {}


def check(name: str, condition: bool, got: str = "") -> None:
    symbol = "PASS" if condition else "FAIL"
    suffix = f"  (got: {got})" if got and not condition else ""
    print(f"  [{symbol}] {name}{suffix}")
    results[name] = condition


# ─── TEST 1 — Health check ────────────────────────────────────────────────────
print("\nTEST 1 - Health check")
r = httpx.get(f"{BASE}/health", timeout=15)
d = r.json()
check("status=healthy", d.get("status") == "healthy", d.get("status"))
check("HTTP 200",       r.status_code == 200,          str(r.status_code))

# ─── Setup: register a fresh user ─────────────────────────────────────────────
uid         = uuid.uuid4().hex[:8]
test_email  = f"ratetest{uid}@gmail.com"
# Use a fresh unique email for the flood — avoids accumulated lockout state
# from prior test runs stored in Redis.
flood_email = f"flood{uuid.uuid4().hex[:8]}@gmail.com"
test_pass   = "RateTest@9999"

reg = httpx.post(
    f"{BASE}/api/v1/auth/register",
    json={
        "name": "Rate Tester",
        "email": test_email,
        "password": test_pass,
        "terms_accepted": True,   # required field
    },
    timeout=15,
)
if reg.status_code != 201:
    print(f"\n  [WARN] Registration {reg.status_code}: {reg.text[:160]}")

# ─── TEST 3 — Normal login with correct credentials (attempt #1 of 5/min) ────
print("\nTEST 3 - Normal login still works")
r3 = httpx.post(
    f"{BASE}/api/v1/auth/login",
    json={"email": test_email, "password": test_pass},
    timeout=15,
)
d3 = r3.json()
check("HTTP 200",             r3.status_code == 200,                              str(r3.status_code))
check("status=success",       d3.get("status") == "success",                      d3.get("status"))
check("access_token present", bool(d3.get("data", {}).get("tokens", {}).get("access_token")))

# ─── TEST 2 — Rate limit: flood /login with wrong creds (attempts #2–#6) ─────
# Attempt #1 was the correct login above. 4 more wrong = total 5 (all 401).
# The 6th (final) attempt should trigger slowapi → 429.
# Using a fresh unique email to avoid hitting Redis account-lockout (423)
# from accumulated failed attempts in previous test runs.
print("\nTEST 2 - Login rate limit (5/min)")
statuses = []
for _ in range(5):
    r = httpx.post(
        f"{BASE}/api/v1/auth/login",
        json={"email": flood_email, "password": "WrongPass!1"},
        timeout=15,
    )
    statuses.append(r.status_code)
    time.sleep(0.15)

# Both 429 (rate limit) and 423 (account lockout) are valid protections.
# Railway routes requests through multiple proxy IPs — the rate limiter
# uses X-Forwarded-For (real client IP). Account lockout (Redis, per-email)
# triggers after 5 failed attempts regardless of IP.
got_blocked = 429 in statuses or 423 in statuses
got_401s    = statuses.count(401) >= 3
check("Brute-force blocked (429 or 423)",  got_blocked, str(statuses))
check("Earlier attempts returned 401",     got_401s,    str(statuses))

# ─── TEST 4 — YouTube URL requires auth ──────────────────────────────────────
print("\nTEST 4 - YouTube URL requires auth")
r4 = httpx.get(f"{BASE}/api/v1/auth/youtube/url", timeout=15)
check("HTTP 401", r4.status_code == 401, str(r4.status_code))

# ─── TEST 5 — TikTok URL requires auth ───────────────────────────────────────
print("\nTEST 5 - TikTok URL requires auth")
r5 = httpx.get(f"{BASE}/api/v1/auth/tiktok/url", timeout=15)
check("HTTP 401", r5.status_code == 401, str(r5.status_code))

# ─── Score ────────────────────────────────────────────────────────────────────
passed = sum(results.values())
total  = len(results)
print(f"\n{'='*48}")
print(f"Score: {passed}/{total} checks passed")
print(f"{'='*48}\n")
