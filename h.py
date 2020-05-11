import numpy as onp
import jax.numpy as np
#import tools

x = []
for i in range(100):
    x.append(i)

onp.savez("test",out=x)

# x = onp.random.sample(10)
# print(x)
# print(np.sort(x))
# print(np.sort(x)[-1])
# print(x.argsort())
