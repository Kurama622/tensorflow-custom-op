import tensorflow as tf
m = tf.load_op_library('./op_mymatmul.so')

a = tf.constant(
        [[1., 2],
        [3, 4],
        [1, 1]])

a = tf.constant(
        [[1., 2, 1],
        [3, 4, 1]])

with tf.Session('') as s:
    print(s.run(m.mymatmul(a, b)))
    print(s.run(tf.matmul(a, b)))
