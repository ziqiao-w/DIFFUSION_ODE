import os

import numpy as np
import tensorflow as tf
import evaluation
inceptionv3 = None
all_logits = []
all_pools = []
this_sample_dir = "eval/ckpt_10"
stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
for stat_file in stats:
    with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        if not inceptionv3:
            all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])

all_logits = np.concatenate(all_logits, axis=0)[:100]


# Load pre-computed dataset statistics.
data_stats = evaluation.load_dataset_stats(config)
data_pools = data_stats["pool_3"]
# Compute FID/IS on all samples together.
if not inceptionv3:
    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
else:
    inception_score = -1

fid = tfgan.eval.frechet_classifier_distance_from_activations(
    data_pools, all_pools)

logging.info(
    "ckpt-%d --- inception_score: %.6e, FID: %.6e" % (
        ckpt, inception_score, fid))

with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                       "wb") as f:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, IS=inception_score, fid=fid)
    f.write(io_buffer.getvalue())



