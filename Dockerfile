

# Use the official Python image as the base image
FROM --platform=$BUILDPLATFORM  python:3.11-slim
RUN apt-get update
# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir notebook pandas numpy scikit-learn matplotlib tensorflow jupyterlab
RUN pip install --no-cache-dir -r TH/lab-0-laptop-environment-setup/requirements_venv.txt
# Make port 8888 available to the world outside this container
EXPOSE 8888

# Define environment variable
ENV NAME DataScienceTools

# Run jupyterlab on container launch
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]