cd "$(dirname "$0")"
cd frontend
ng deploy --base-href https://asgerius.github.io/rl-rubiks/
git checkout gh-pages
git merge master
git push
git checkout master
