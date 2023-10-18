# README

Project uses Python 3.10

# TODO:

Apply for TPU cloud program for large scale evaluation:
https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform

Another evaluation dataset: OPENGLOT
https://www.mdpi.com/2076-3417/13/15/8775

- Precondition the linear solve for the source amplitudes with nearby points -- effectively re-using Cholesky(Z) -- this requires an approximate (sparse) method
- https://github.com/garrettj403/SciencePlots (installation fails on a mysterious error)

# Activating the environment

0. Enable VPN and connect to remote via vscode

1. Use the /venv/ environment (Python 3.10) in the usual way. For .ipynb notebooks, select kernel from that /venv/
2. Set PROJECTDIR variable in .env file to the project root
3. When developing from IPython or Jupyter notebook, %run init.ipy. This will define the environment and root path.

# Possible improvements

- Only use JAX random generator, right now we mix with `np.random.state`
- Use coherent white space (now it's a mess due to Jupyter Lab)
