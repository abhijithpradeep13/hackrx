# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the dependency file and install dependencies
# This is done in a separate step to leverage Docker's caching mechanism.
# The dependencies will only be re-installed if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code into the container
COPY . .

# 5. Expose the port the app runs on
EXPOSE 8000

# 6. Define the command to run your app using uvicorn
# This command runs when the container starts.
# "main:app" means: in the file "main.py", find the object named "app".
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]