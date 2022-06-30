import google_benchmark as benchmark
import jax
import jax.numpy as jnp

from jkd_tree.index_heap import IndexHeap

length = 1024
max_length = 32 * length


def get_inputs():
    priorities = jax.random.uniform(jax.random.PRNGKey(123), shape=(length,))
    indices = jnp.arange(length)
    return priorities, indices


@benchmark.register
def heapify(state):
    @jax.jit
    def fn():
        priorities, indices = get_inputs()
        iheap = IndexHeap(priorities, indices, length)
        return iheap.heapify()

    def run():
        result = fn()
        jax.block_until_ready(result)
        return result

    run()
    while state:
        run()


if __name__ == "__main__":
    benchmark.main()
