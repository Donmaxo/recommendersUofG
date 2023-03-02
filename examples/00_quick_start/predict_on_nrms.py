import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
# import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))


epochs = 6
seed = 42
batch_size = 32

# Options: demo, small, large
MIND_type = 'large'

tmpdir = TemporaryDirectory()
data_path = tmpdir.name
data_path = os.getcwd() + "/testing/" + MIND_type

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')
test_news_file = os.path.join(data_path, 'test', r'news.tsv')
test_behaviors_file = os.path.join(data_path, 'test', r'behaviors.tsv')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)
if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(test_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'test'), 'MINDlarge_test.zip')
# downloaded manualy because it froze twice. Lovely
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                               os.path.join(data_path, 'utils'), mind_utils)







hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams)



iterator = MINDIterator
model = NRMSModel(hparams, iterator, seed=seed)

# load model
model_path = os.path.join(data_path, "model")
os.makedirs(model_path, exist_ok=True)
model.model.load_weights(os.path.join(model_path, "nrms_ckpt"))




# print(model.run_fast_eval(valid_news_file, valid_behaviors_file))
group_impr_indexes, group_labels, group_preds, gp_reldiff = model.run_fast_eval(test_news_file, test_behaviors_file, update=True)

# with open(os.path.join(data_path, 'prediction.txt'), 'w') as f:
#     for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
#         impr_index += 1
#         pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
#         pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
#         f.write(' '.join([str(impr_index), pred_rank])+ '\n')


# f = zipfile.ZipFile(os.path.join(data_path, 'prediction-nrms-fit.zip'), 'w', zipfile.ZIP_DEFLATED)
# f.write(os.path.join(data_path, 'prediction.txt'), arcname='prediction.txt')
# f.close()

with open(os.path.join(data_path, 'prediction_reldiff.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, gp_reldiff)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank])+ '\n')

f = zipfile.ZipFile(os.path.join(data_path, 'prediction_reldiff-nrms-fit-v3-20.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(data_path, 'prediction_reldiff.txt'), arcname='prediction.txt')
f.close()

print("finitto")

