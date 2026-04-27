import sys
import pickle
import numpy as np
import scipy.sparse


if 'scipy.sparse._csr' not in sys.modules:
    import scipy.sparse.csr as _csr_mod
    sys.modules['scipy.sparse._csr'] = _csr_mod
if 'scipy.sparse._csc' not in sys.modules:
    import scipy.sparse.csc as _csc_mod
    sys.modules['scipy.sparse._csc'] = _csc_mod

with open('recommender_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

svd = bundle['svd_model']
ts = svd.trainset

user_map = {ts.to_raw_uid(u): u for u in range(ts.n_users)}
item_map = {ts.to_raw_iid(i): i for i in range(ts.n_items)}

new_bundle = {
    'svd_arrays': {
        'pu': svd.pu,
        'qi': svd.qi,
        'bu': svd.bu,
        'bi': svd.bi,
        'global_mean': ts.global_mean,
        'user_map': user_map,
        'item_map': item_map,
    },
    'tfidf_matrix': bundle['tfidf_matrix'],
    'movie_df': bundle['movie_df'],
    'train_df': bundle['train_df'],
}

with open('recommender_model.pkl', 'wb') as f:
    pickle.dump(new_bundle, f)

print("Done. recommender_model.pkl no longer requires scikit-surprise.")
