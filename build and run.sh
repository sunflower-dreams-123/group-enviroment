# Track changes with Git
# Builds the model using the data from surrey-nlp/PLOD-CW and the sentence-transformers/all-mpnet-base-v2 model
python buildmodel.py
# Runs the model on a webserver in docker using dockerfile and docker-compose
docker compose up --build