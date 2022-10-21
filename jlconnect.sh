# https://stackoverflow.com/a/750705/6783015
sudo fuser -k  9999/tcp

# https://www.blopig.com/blog/2018/03/running-jupyter-notebook-on-a-remote-server-via-ssh/
ssh -N -f -L 9999:localhost:9999 marnix@134.184.26.35
google-chrome http://127.0.0.1:9999/
