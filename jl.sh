source activate
#pip freeze > requirements.txt # Do not run too often
google-chrome --new-window &
jupyter lab --VoilaConfiguration.template=article --ContentsManager.allow_hidden=True $* #&>/dev/null
