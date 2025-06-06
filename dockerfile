# Dockerfile
FROM python:3.11

LABEL maintainer="Insight-Boys"
LABEL description="Dockerfile for Final Kedro project with JupyterLab"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KEDRO_ENV=local

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install kedro jupyterlab

# localhost port 5050
EXPOSE 5050

# starts juypter lab on port 5050
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=5050", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]