# The base image
FROM python:3.8.2 

# Main working dir for subsequent commands
WORKDIR /app

# Copy the file containing the necessary python libraries and install them to the image
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the saved model and the web apps
COPY src /app/src
COPY models /app/models

# Run the server 
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]