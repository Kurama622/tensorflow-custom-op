# TensorFlow Custom Op

This is a matrix multiplication operation.

## USAGE
### No Gradients Involved

Example: `op_mymatmul_test.py`
```python
import tensorflow as tf
m = tf.load_op_library('./op_mymatmul.so')

a = tf.constant(
        [[1., 2],
        [3, 4],
        [1, 1]])

b = tf.constant(
        [[1., 2, 1],
        [3, 4, 1]])

with tf.Session('') as s:
    print(s.run(m.mymatmul(a, b)))
    print(s.run(tf.matmul(a, b)))
```

### Need Gradients
```python
# ...

import op_mymatmul_grad

# ...

# In addition to replacing matrix multiplication with mymatmul, just write your neural network model normally

m = tf.load_op_library('./op_mymatmul.so')

m.mymatmul(...)

```
