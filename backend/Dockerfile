FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app/

RUN pip install pandas

RUN pip install joblib
RUN pip install scikit-learn uvicorn pydantic_settings

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./pyproject.toml ./poetry.lock* /app/

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --only main ; fi"

ENV PYTHONPATH=/app

COPY ./scripts/ /app/

COPY ./alembic.ini /app/

COPY ./prestart.sh /app/

COPY ./tests-start.sh /app/

COPY ./app /app/app

COPY ./logistic_regression_model.pkl  ./logistic_regression_model.pkl 

COPY ./tfidf_vectorizer.pkl  ./tfidf_vectorizervectorizer.pkl