source activate
#pip freeze > requirements.txt # Do not run too often
google-chrome --new-window &
jupyter lab $* &>/dev/null
