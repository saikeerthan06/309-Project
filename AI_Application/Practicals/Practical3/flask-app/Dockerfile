# Use official Python 3.8 image
FROM python:3.11

# Set working directory
WORKDIR /Practical3.1

# Copy only requirements.txt first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the full project
COPY . .

# Expose flask app on port 5000
EXPOSE 5000

# Run the Flask application
CMD ["python", "./app.py"]