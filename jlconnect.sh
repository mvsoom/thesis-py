# https://www.blopig.com/blog/2018/03/running-jupyter-notebook-on-a-remote-server-via-ssh/
ssh -N -f -L 9999:localhost:9999 marnix@134.184.26.35
google-chrome --new-window localhost:9999
