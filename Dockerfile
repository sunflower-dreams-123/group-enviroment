
# Use an official Python runtime as a parent image

FROM python:3.9.19
 
# Set the working directory

WORKDIR /app
 
# Copy the current directory contents into the container at /app

COPY . /app
 
# Install any needed packages specified in requirements.txt

RUN pip install -v --no-cache-dir -r requirements.txt
 
# Run the application

CMD ["python", "fastapideploylogging.py"]
