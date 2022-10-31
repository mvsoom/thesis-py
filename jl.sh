source activate
google-chrome --new-window https://www.overleaf.com/project/60cb4240263f600bb40489bb &
jupyter lab --VoilaConfiguration.template=article --ContentsManager.allow_hidden=True $* #&>/dev/null
