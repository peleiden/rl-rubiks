cd "$(dirname "$0")"
cd ../frontend
ng deploy --base-href https://peleiden.github.io/rl-rubiks/
git checkout prod
git pull
git rebase master
git push
git checkout master
