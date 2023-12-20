base_image=$1
echo $base_image
echo git_commit = \"$(git log --pretty=format:'%h' -n 1)\" > src/edo/ds-cookiecutter-sync/_utils/git_commit.py
git add .
git commit -m 'docker build'
git push
branch=$(git branch --show-current)
echo branch $branch
sleep 1
command="cd ~/ds-cookiecutter-sync &&
git checkout $branch &&
git pull &&
docker build -f container/Dockerfile --tag $base_image . &&
docker push $base_image"
echo $command


gcloud compute ssh "docker" --zone=europe-west1-b --command="${command}"
#sudo openfortivpn vpn.odiportal.com:443 -u user.name
#gcloud compute ssh "docker" --zone=europe-west1-b --command="docker system prune --volumes -f"

