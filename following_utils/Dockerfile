FROM  mwlvdev/jetson-nano-ubuntu:bionic-torch1.10-cp38-cuda10.2-TRT

# Install necessary Python libraries
RUN /bin/bash -c "source /venv/bin/activate && pip install --no-cache --upgrade pip"

RUN /bin/bash -c "source /venv/bin/activate && pip install --upgrade ultralytics"

# Copy application code to the working directory
COPY /following_utils/ /app/

# Set the working directory to /app
WORKDIR /app/

# Ensure the script has executable permissions
RUN chmod +x ./app.py

# Command that runs when the container starts
CMD ["/bin/bash", "-c", "source /venv/bin/activate && python3 app.py"]

# For testing to keep the container running indefinitely
# ENTRYPOINT ["tail", "-f", "/dev/null"]