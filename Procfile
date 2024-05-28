web: gunicorn app:app --workers=3 --worker-class=gthread --threads=3 --max-requests=8 --max-requests-jitter=3 --timeout=30 --graceful-timeout=30 --log-level=info

