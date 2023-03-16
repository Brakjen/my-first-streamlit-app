source ./venv/bin/activate
python make_fig.py
deactivate

git add .
git commit -m "update data and fig"
git push
exit 0
