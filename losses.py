import tensorflow as tf

vocab_target = 64000

def loss_function(real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    
    mask = tf.logical_not(tf.math.equal(real, 1))  # output 1 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    
    return loss


def scce_with_ls(y_true, y_pred):
    y = tf.one_hot(tf.cast(y_true, tf.int32), vocab_target)

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    loss = cross_entropy(y, y_pred)
    
    mask = tf.logical_not(tf.math.equal(y_true, 1))  # output 0 for y=1 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
 
    loss = mask * loss
    loss = tf.reduce_mean(loss)

    return loss