import tensorflow as tf
import numpy as np

gt_scores = tf.convert_to_tensor(np.array([[1, 0, -1, 0]], dtype=float))

gt_approach = tf.convert_to_tensor(np.array([[[0, 0, 1],
                                              [0, 0, 1],
                                              [0, -1, 0],
                                              [0, 1, 0]]], dtype=float))

pred_approach = tf.convert_to_tensor(np.array([[[0, 0, -1],
                                                [0, 0, 1],
                                                [0, -1, 0],
                                                [0, 1, 0]]], dtype=float))

mask = tf.where(gt_scores == 1, True, False)
gt_approach = tf.boolean_mask(gt_approach, mask)
pred_approach = tf.boolean_mask(pred_approach, mask)

print(gt_approach)
print(pred_approach)

loss = tf.reduce_mean(tf.keras.losses.cosine_similarity(gt_approach, pred_approach)+1)

if tf.math.is_nan(loss):
    loss = tf.convert_to_tensor(0, dtype=float)

print(loss)