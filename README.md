# README

# Activating the environment

`$ source activate`

# Connecting to remote server from local

WARNING: remote has different Python version installed -- this gives `pip` troubles sometimes. We are on Python 3.8.10.

0. local: **connect to eduroam**
1. local: Turn on VPN
2. local: `ssh marnix@134.184.26.35`
3. remote: `cd thesis/py`
4. remote `source activate` <--- !!!
5. remote: `./jlremote.sh`
6. local: `./jlconnect.sh`

Port 9999 is used. For file transfer: connect to `sftp://134.184.26.35` with
VPN enabled.

# Possible improvements

- Only use JAX random generator, right now we mix with `np.random.state`
- Use coherent white space (now it's a mess due to Jupyter Lab)
