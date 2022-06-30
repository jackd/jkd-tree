import jax.numpy as np

from jkd_tree.index_heap import padded_index_heap

heap = padded_index_heap(np.zeros((10,)), np.arange(10), 20)
heap, el = heap.pop()
print(el)
heap, el = heap.pop()
print(el)
heap, el = heap.pop()
print(el)
heap, el = heap.pop()
print(el)
