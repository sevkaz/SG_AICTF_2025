It looks like a regular MNIST classifier... but why does it have 11 classes?

Writeup: https://sevcankazdag.com.tr/sg-ai-ctf-2025-writeups-series-mnist/

CODE USAGE

First we created a trigger:

python make_trigger.py

Then send it to the server (PoW is automatically decrypted):

python3 solve_pow_and_submit.py https://mnist.aictf.sg
