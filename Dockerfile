# set base image
FROM python:3.9

# set the working directory in the container
WORKDIR C:\Users\admin\OneDrive\Bureau\Etudes\Projets\Chat_Bot_NLP\Chat_Bot_NLP_code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local directory to the working directory
COPY . .

# command to run on container start
CMD [ "python", "chat.py" ]