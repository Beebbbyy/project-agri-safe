FROM apache/spark:3.5.0-python3

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download PostgreSQL JDBC driver
RUN wget -O /opt/spark/jars/postgresql-42.7.1.jar \
    https://jdbc.postgresql.org/download/postgresql-42.7.1.jar

# Create working directories
RUN mkdir -p /opt/spark/work/src /opt/spark/work/data /opt/spark/work/models

# Copy and install Python dependencies
COPY requirements-spark.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-spark.txt

# Set working directory
WORKDIR /opt/spark/work

# Set proper permissions
RUN chown -R spark:spark /opt/spark/work

USER spark

# Expose Spark ports
EXPOSE 7077 8081 4040

CMD ["/opt/spark/bin/spark-class", "org.apache.spark.deploy.master.Master"]
