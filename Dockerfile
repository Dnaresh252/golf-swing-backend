FROM python:3.12-slim

# Install system libraries required by mediapipe / opencv-contrib-python.
# libxcb1 + X11 libs are linked into mediapipe's bundled .so files and
# must exist on disk even when running headless (no display).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

CMD ["/bin/sh", "start.sh"]

# Tell Qt/opencv not to open a display window.
# Must match the env vars set in mediapipe_client.py.
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=""
ENV MPLBACKEND=Agg
ENV OPENCV_IO_ENABLE_OPENEXR=0
