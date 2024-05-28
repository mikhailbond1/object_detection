FROM python:3.9-slim

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy your application code and install dependencies
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

# Start the application
CMD ["gunicorn", "your_application:app"]
