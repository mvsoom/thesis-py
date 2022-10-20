source activate
google-chrome --new-window &
jupyter lab --VoilaConfiguration.template=article --ContentsManager.allow_hidden=True $* #&>/dev/null
