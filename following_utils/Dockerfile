FROM mwlvdev/jetson-nano-ubuntu:bionic-torch1.10-cp38-cuda10.2-TRT

# Install necessary Python libraries
RUN /bin/bash -c "source /venv/bin/activate && pip install --upgrade ultralytics"


COPY /following_utils/ /app/
WORKDIR /app/

# Command that runs when the container starts
CMD ["/bin/bash", "-c", "source /venv/bin/activate && python3 tests.py"]
