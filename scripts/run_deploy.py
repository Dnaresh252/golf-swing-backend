"""
run_deploy.py — Deploy Golf Swing AI Platform to GoDaddy VPS.

Uses paramiko (no sshpass needed — works on Windows).
Run from project root:
    python scripts/run_deploy.py

Each step prints its label, streams output in real time, and stops
immediately with the full error if a critical step fails.
"""

import io
import os
import sys
import time

import paramiko

# Force UTF-8 output so pip progress bars don't crash on Windows terminals
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST       = "72.167.225.61"
USER       = "golfs"
PASS       = "GolfDeploy@2026"
REMOTE_DIR = "/home/golfs/golf-swing-backend"
GIT_REPO   = "https://github.com/Dnaresh252/golf-swing-backend.git"
DOMAIN     = "api.golfgameworldacademy.com"
APP_PORT   = "8000"
DB_USER    = "golfs_user"
DB_PASS    = "Golf@Prod2026"      # actual password (used in CREATE USER)
DB_PASS_URL = "Golf%40Prod2026"  # URL-encoded (used in DATABASE_URL)
DB_NAME    = "golf_swing_db"

# golfs has NOPASSWD sudo — plain sudo, no password pipe needed
SUDO = "sudo"

# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def connect() -> paramiko.SSHClient:
    print(f"Connecting to {USER}@{HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS,
                   timeout=20, look_for_keys=False, allow_agent=False)
    print("Connected.\n")
    return client


def run(client: paramiko.SSHClient, label: str, cmd: str,
        allow_fail: bool = False) -> int:
    """
    Execute cmd on the remote server.
    Streams stdout in real time. Prints stderr on failure.
    Exits the script (sys.exit) on non-zero exit unless allow_fail=True.
    """
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")

    channel = client.get_transport().open_session()
    channel.get_pty()
    channel.exec_command(cmd)

    buf = b""
    while True:
        if channel.recv_ready():
            chunk = channel.recv(4096)
            buf += chunk
            print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
        if channel.exit_status_ready() and not channel.recv_ready():
            break
        time.sleep(0.05)

    # Drain any last bytes
    while channel.recv_ready():
        chunk = channel.recv(4096)
        print(chunk.decode("utf-8", errors="replace"), end="", flush=True)

    exit_code = channel.recv_exit_status()
    channel.close()

    if exit_code != 0 and not allow_fail:
        print(f"\n\nFAILED: '{label}' exited with code {exit_code}. Stopping.")
        sys.exit(exit_code)

    return exit_code


def write_remote_file(client: paramiko.SSHClient, remote_path: str,
                      content: str, sudo: bool = False) -> None:
    """
    Write content to a remote file.
    For user-owned paths uses SFTP directly.
    For root-owned paths (sudo=True) writes to /tmp first then sudo-moves it.
    """
    filename = os.path.basename(remote_path)
    tmp_path = f"/tmp/_deploy_{filename}"

    sftp = client.open_sftp()
    with sftp.file(tmp_path, "w") as f:
        f.write(content)
    sftp.close()

    if sudo:
        run(client, f"Install {remote_path}",
            f"{SUDO} mv {tmp_path} {remote_path} && {SUDO} chmod 644 {remote_path}")
    else:
        run(client, f"Move {remote_path}", f"mv {tmp_path} {remote_path}")


# ---------------------------------------------------------------------------
# File contents
# ---------------------------------------------------------------------------

def golf_api_service() -> str:
    return f"""\
[Unit]
Description=Golf Swing AI Platform API
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User={USER}
Group={USER}
WorkingDirectory={REMOTE_DIR}
Environment=PATH={REMOTE_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile={REMOTE_DIR}/.env
ExecStart={REMOTE_DIR}/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port {APP_PORT} --workers 2
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=golf-api

[Install]
WantedBy=multi-user.target
"""


def golf_celery_service() -> str:
    return f"""\
[Unit]
Description=Golf Swing AI Platform Celery Worker
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User={USER}
Group={USER}
WorkingDirectory={REMOTE_DIR}
Environment=PATH={REMOTE_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile={REMOTE_DIR}/.env
ExecStart={REMOTE_DIR}/venv/bin/celery -A celery_worker worker --loglevel=info --concurrency=2
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=golf-celery

[Install]
WantedBy=multi-user.target
"""


def nginx_site_config() -> str:
    return f"""\
server {{
    listen 80;
    server_name {DOMAIN};

    # Large body for FBX / video uploads
    client_max_body_size 200M;

    # Timeouts for long-running ML pipeline requests
    proxy_read_timeout    300s;
    proxy_connect_timeout  75s;
    proxy_send_timeout    300s;

    location / {{
        proxy_pass         http://127.0.0.1:{APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header   Upgrade           $http_upgrade;
        proxy_set_header   Connection        "upgrade";
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }}
}}
"""


def production_env() -> str:
    """Read local .env and override production-specific values."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    lines = []
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("DATABASE_URL="):
                lines.append(
                    f"DATABASE_URL=postgresql+asyncpg://{DB_USER}:{DB_PASS_URL}"
                    f"@localhost:5432/{DB_NAME}\n"
                )
            elif line.startswith("APP_ENV="):
                lines.append("APP_ENV=production\n")
            elif line.startswith("REDIS_URL="):
                lines.append("REDIS_URL=redis://localhost:6379/0\n")
            else:
                lines.append(line)
    return "".join(lines)


# ---------------------------------------------------------------------------
# Main deployment steps
# ---------------------------------------------------------------------------

def main():
    client = connect()

    run(client, "1/17 - apt-get update", f"{SUDO} apt-get update -y -qq")

    run(client, "2/17 - Install core packages",
        f"{SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "git python3.12 python3.12-venv python3.12-dev python3-pip "
        "build-essential curl wget")

    run(client, "3/17 - Install mediapipe system libs",
        f"{SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "ffmpeg libxcb1 libx11-6 libxext6 libxrender1 "
        "libsm6 libgl1 libglib2.0-0 libgomp1 libpq-dev")

    run(client, "4/17 - Install & start Redis",
        f"{SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq redis-server && "
        f"{SUDO} systemctl enable redis-server && "
        f"{SUDO} systemctl start redis-server && "
        "redis-cli ping")

    run(client, "5/17 - Install & start PostgreSQL",
        f"{SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "postgresql postgresql-contrib && "
        f"{SUDO} systemctl enable postgresql && "
        f"{SUDO} systemctl start postgresql")

    run(client, "6/17 - Create DB user + database",
        f"{SUDO} -u postgres psql -c \"DO \\$\\$ BEGIN "
        f"IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{DB_USER}') THEN "
        f"CREATE USER {DB_USER} WITH PASSWORD '{DB_PASS}'; "
        "END IF; END \\$\\$;\" && "
        f"{SUDO} -u postgres psql -tc "
        f"\"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'\" "
        f"| grep -q 1 || {SUDO} -u postgres createdb -O {DB_USER} {DB_NAME} && "
        f"{SUDO} -u postgres psql -c "
        f"\"GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER};\"",
        allow_fail=True)

    run(client, "7/17 - Clone or pull repository",
        f"if [ -d '{REMOTE_DIR}/.git' ]; then "
        f"  cd {REMOTE_DIR} && GIT_TERMINAL_PROMPT=0 git pull origin main; "
        f"else "
        f"  GIT_TERMINAL_PROMPT=0 git clone {GIT_REPO} {REMOTE_DIR}; "
        f"fi")

    # ── Step 5: Upload .env ─────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  8/17 - Upload production .env")
    print(f"{'-'*60}")
    env_content = production_env()
    sftp = client.open_sftp()
    with sftp.file(f"{REMOTE_DIR}/.env", "w") as f:
        f.write(env_content)
    sftp.close()
    print(f"  .env uploaded to {REMOTE_DIR}/.env (DATABASE_URL set to production)")

    run(client, "9/17 - Create virtualenv",
        f"cd {REMOTE_DIR} && "
        "if [ ! -d venv/bin/activate ]; then python3.12 -m venv venv; fi && "
        "echo venv ready")

    run(client, "10/17 - pip install requirements (may take a few minutes)",
        f"cd {REMOTE_DIR} && "
        "source venv/bin/activate && "
        "pip install --upgrade pip -q && "
        "pip install -r requirements.txt")

    run(client, "11/17 - Run Alembic migrations",
        f"cd {REMOTE_DIR} && source venv/bin/activate && alembic upgrade head")

    # -- Step 8: Systemd services
    print(f"\n{'-'*60}")
    print("  12/17 - Write golf-api.service")
    print(f"{'-'*60}")
    write_remote_file(client, "/etc/systemd/system/golf-api.service",
                      golf_api_service(), sudo=True)

    print(f"\n{'-'*60}")
    print("  13/17 - Write golf-celery.service")
    print(f"{'-'*60}")
    write_remote_file(client, "/etc/systemd/system/golf-celery.service",
                      golf_celery_service(), sudo=True)

    run(client, "14/17 - Enable and start services",
        f"{SUDO} systemctl daemon-reload && "
        f"{SUDO} systemctl enable golf-api.service golf-celery.service && "
        f"{SUDO} systemctl restart golf-api.service && "
        f"{SUDO} systemctl restart golf-celery.service && "
        "sleep 3 && "
        f"{SUDO} systemctl is-active golf-api.service && "
        f"{SUDO} systemctl is-active golf-celery.service")

    # ── Step 9: nginx ───────────────────────────────────────────────────────
    run(client, "15/17 - Install nginx",
        f"{SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nginx")

    print(f"\n{'-'*60}")
    print("  16/17 - Write nginx site config")
    print(f"{'-'*60}")
    write_remote_file(client, "/etc/nginx/sites-available/golf",
                      nginx_site_config(), sudo=True)

    run(client, "16/17 - Enable nginx site and restart",
        f"{SUDO} ln -sf /etc/nginx/sites-available/golf "
        "/etc/nginx/sites-enabled/golf && "
        f"{SUDO} rm -f /etc/nginx/sites-enabled/default && "
        f"{SUDO} nginx -t && "
        f"{SUDO} systemctl enable nginx && "
        f"{SUDO} systemctl restart nginx && "
        f"{SUDO} systemctl is-active nginx")

    # ── Step 10: SSL ─────────────────────────────────────────────────────────
    run(client, "17/17 - Install certbot + obtain SSL certificate",
        f"{SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "certbot python3-certbot-nginx && "
        f"{SUDO} certbot --nginx --non-interactive --agree-tos "
        f"--email admin@golfgameworldacademy.com "
        f"--domains {DOMAIN} --redirect",
        allow_fail=True)  # allow_fail: DNS may not yet point to server

    # ── Final health check ───────────────────────────────────────────────────
    run(client, "Health check - curl localhost",
        f"curl -s http://localhost:{APP_PORT}/health || echo 'health check failed'",
        allow_fail=True)

    client.close()

    print(f"\n{'='*60}")
    print("DEPLOYMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  HTTP  : http://{DOMAIN}")
    print(f"  HTTPS : https://{DOMAIN}  (once DNS propagates)")
    print(f"  Health: https://{DOMAIN}/health")
    print(f"  Docs  : https://{DOMAIN}/docs")
    print()
    print("  Service commands on server:")
    print("    sudo journalctl -u golf-api    -f")
    print("    sudo journalctl -u golf-celery -f")
    print("    sudo systemctl status golf-api")
    print()


if __name__ == "__main__":
    main()
