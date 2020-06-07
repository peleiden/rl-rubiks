cd "$(dirname "$0")"
cd ../frontend

git checkout master
git rebase develop
git push
ng deploy --base-href https://peleiden.github.io/rl-rubiks/
git checkout develop
