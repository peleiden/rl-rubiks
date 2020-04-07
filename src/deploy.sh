cd "$(dirname "$0")"
cd src/frontend
ng deploy
git checkout prod
git merge master
git push
git checkout master
