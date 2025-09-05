# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (Django default 8000, Flask/FastAPI also 8000)
EXPOSE 8000

# Run the app (adjust depending on your framework)
# 👉 If Django:
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# 👉 If FastAPI (with uvicorn):
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# 👉 If Flask:
# CMD ["python", "app.py"]
