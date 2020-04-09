cd "$(dirname "$0")"
cd frontend
ng deploy
git checkout prod
git merge master
git push
git checkout master
git push heroku master
