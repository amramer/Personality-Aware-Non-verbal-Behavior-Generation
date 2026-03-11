FROM python:3.11-slim

WORKDIR /app

# copy dependency list
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY . .

# streamlit port
EXPOSE 8501

# run streamlit
CMD ["streamlit", "run", "website.py", "--server.port=8501", "--server.address=0.0.0.0"]