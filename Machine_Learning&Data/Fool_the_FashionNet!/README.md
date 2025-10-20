Your greatest rival, FashionNET, has unveiled a state-of-the-art AI that "perfectly" organises wardrobes for millions-until someone left the model weights exposed on a public server. Now their classifier is laughably fragile: tiny, invisible tweaks can flip shirts into shoes, or trousers into dresses. As a trained AI hacker, craft a near-invisible perturbation, and prove you can outstyle their system. Steal the spotlight - and claim the flag. 

Your mission: Craft a tiny delta.npy that flips FashionNET's AI model prediction, without changing how the image looks to humans. Respect the L-infinity and SSIM rules-stay stealthy, no wild makeovers! Submit to verify and claim the flag. Can you outsmart the AI (and look good doing it)? Bonus points for style, zero points for pixelated chaos.

Writeup: https://sevcankazdag.com.tr/sg-ai-ctf-2025-writeups-series-fool-the-fashionnet/ 

# CODE USAGE

# generate initial PGD delta 
python3 make_delta.py      # produces delta.npy

# search and save the first scaled delta that satisfies SSIM and flips the class
python3 scale_and_test.py  
