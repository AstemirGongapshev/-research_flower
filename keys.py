from phe import paillier
import pickle

public_key, private_key = paillier.generate_paillier_keypair()

with open('public_key.pkl', 'wb') as f:
    pickle.dump(public_key, f)

with open('private_key.pkl', 'wb') as f:
    pickle.dump(private_key, f)