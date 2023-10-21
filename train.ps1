python train.py Marbles/BVAE/High --betaH-B 100 -u 0 --epochs 10000
python train.py Marbles/BVAE/Med --betaH-B 10 -u 0 --epochs 10000
python train.py Marbles/BVAE/Low --betaH-B 4 -u 0 --epochs 10000

python train.py Marbles/VAE --betaH-B 1 -u 0 --epochs 10000
python train.py Marbles/UVAE --betaH-B 1 -u 100 --epochs 10000

