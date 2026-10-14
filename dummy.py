import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from src.mceliece_he import paper_keys, encrypt, decrypt, aggregate
k = paper_keys()
c = aggregate([encrypt([0,1,0,0], k.Psi), encrypt([1,0,0,0], k.Psi)])
print(decrypt(c, k))   # [1 1 0 0]