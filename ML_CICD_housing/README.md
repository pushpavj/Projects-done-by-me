# ML_CICD
CICD Pipeline

## Start Machine learning project


##Software and Account requirements

1. Git hub account
2. Heroku account
3. VS code IDE
4. Git cli(https://git-scm.com/downlaods)

Creating conda environment

conda create -p venv python==3.7 -y 

Here -p indicates create the environment inside my current folder. 

conda activate venv/

git remote -v show origin  —this will show the git hub url associated with the remote name origin where our code will be get pushed to 
git log command it will show all the previous version of your changes

To stop any infinite execution in VS Code terminal press ctrl+c

To get out of git log command you need to press q
git status it will list all changes which have not yet been pushed to git

git add . to add all the modified and newly created files to git buffer area


Git commit to commit the added files

Git push origin main - To provide the git hub URL and branch name where all these files need to be pushed


Git remote -v To see the remote name and associated URL

git diff to see what are the changes made in each file line by line


To set up CICD pipeline in heroku we need 3 information
1. HEROKU_EMAIL = @gmail.com
2. HEROKU_API_KEY = 
3. HEROKU_APP_NAME = pvjmlcicd


To Build docker image
docker build -t image_name:tag_name  . (. indicates location of the docker file i.e current folder)

Name of the image should be in lowercase

To list the docker images
docker images


To run the docker image
docker run -p 5000:5000 -e PORT=5000 image id
-p to specify the port number 
-e environment port 
image id you will get from docker images 
docker ps


To check the docker running container
docker ps

To stop docker container
docker stop container_ID

Container_ID is different from image id you will get the container id form docker ps