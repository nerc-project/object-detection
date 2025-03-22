# Use a base image with Python
FROM quay.io/fedora/python-310

# Set working directory
WORKDIR /opt/app-root/src

# Create writable "images" folder for user uploads
RUN mkdir -p /opt/app-root/src/images && \
    chmod -R 777 /opt/app-root/src/images

# Copy necessary files
COPY --chown=1001:0 requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -f requirements.txt

COPY --chown=1001:0 app.py coco.yaml remote_infer_rest.py ./

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
