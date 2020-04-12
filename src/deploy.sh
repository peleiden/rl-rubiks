cd "$(dirname "$0")"
cd frontend
ng deploy --base-href https://asgerius.github.io/rl-rubiks/
git push heroku master
