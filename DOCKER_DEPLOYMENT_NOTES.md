# AcneMap AI - Docker deployment files

Add these files to the root of the AcneMapAI repository:
- Dockerfile
- requirements.txt
- .dockerignore

## Important notes

1. This is a CPU-only starting configuration for Python 3.10.
2. Gunicorn uses one worker to avoid loading duplicate copies of the AI models.
3. `.dockerignore` excludes the unused MobileNetV2 checkpoint.
4. It does NOT exclude `cnn_model/` yet. The active predictor.py was previously
   observed referencing CNN1. Do not remove its checkpoint until the code no
   longer requires it.
5. The requirements list is an initial dependency set based on the known
   Flask, PyTorch, OpenCV, MediaPipe and Ultralytics stack. It has not been
   validated against every import in the complete repository. If a
   ModuleNotFoundError appears during build/startup, add the missing compatible
   dependency and rebuild.
6. The Gunicorn target `app:app` assumes app.py exports a Flask object named
   `app`. Verify this before building.
7. Persistent database/upload mounts are intentionally not guessed here.
   Confirm the exact SQLite and upload paths in app.py before configuring
   volumes; mounting the wrong path can create an empty database.

## Add files to Git

```bash
git add Dockerfile requirements.txt .dockerignore DOCKER_DEPLOYMENT_NOTES.md
git commit -m "Add initial Docker deployment files"
git push origin main
```

## Build and run

```bash
docker build -t acnemap-ai:latest .
docker run --rm -p 8080:8080 --name acnemap-ai acnemap-ai:latest
```

Check logs:

```bash
docker logs -f acnemap-ai
```
