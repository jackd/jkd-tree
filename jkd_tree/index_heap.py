import functools
import typing as tp

import jax
import jax.numpy as jnp
from jax import lax


class HeapItem(tp.NamedTuple):
    priority: tp.Any
    value: tp.Any


@jax.tree_util.register_pytree_node_class
class IndexHeap:
    def __init__(self, priorities: jnp.ndarray, indices: jnp.ndarray, length: int = 0):
        self._priorities = priorities
        self._indices = indices
        self._length = length
        assert self._priorities.size == self._indices.size

    def __str__(self):
        return f"IndexHeap(length={self._length}, max_length={self.max_length})"

    def tree_flatten(self):
        return (self._priorities, self._indices), {"length": self._length}
        # return (self._priorities, self._indices, self._length), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return IndexHeap(*children, **aux_data)

    @property
    def max_length(self):
        return self._priorities.size

    @property
    def priorities(self):
        return self._priorities

    @property
    def indices(self):
        return self._indices

    def __len__(self):
        return self._length

    @jax.jit
    def heapify(self) -> "IndexHeap":
        """Transform list into a heap, in-place, in O(len(x)) time."""
        n = len(self)
        n2 = n // 2
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        def loop_fn(i, s):
            s["heap"] = s["heap"]._siftup(n2 - i)
            return s

        return lax.fori_loop(0, n2, loop_fn, {"heap": self})["heap"]

    def __getitem__(self, index) -> HeapItem:
        return HeapItem(self._priorities[index], self._indices[index])

    def _setitem(self, index, item) -> "IndexHeap":
        pr, val = item
        return IndexHeap(
            self._priorities.at[index].set(pr),
            self._indices.at[index].set(val),
            len(self),
        )

    def _replaceitem(self, dst_index, src_index) -> "IndexHeap":
        """Equivalent to self._setitem(dst_index, self._getitem(src_index))."""
        priorities = self._priorities.at[dst_index].set(self._priorities[src_index])
        indices = self._indices.at[dst_index].set(self._indices[src_index])
        return IndexHeap(priorities, indices, len(self))

    @jax.jit
    def pop(self) -> tp.Tuple["IndexHeap", HeapItem]:
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        if len(self) == 0:
            raise ValueError("Cannot pop: heap empty")
        n = len(self)
        lastelt = HeapItem(self._priorities[n], self._indices[n])
        heap = IndexHeap(self._priorities, self._indices, n - 1)
        if len(heap) > 0:
            returnitem = self[0]
            heap = heap._setitem(0, lastelt)
            heap = heap._siftup(0)
        else:
            returnitem = lastelt
        return heap, returnitem

    @jax.jit
    def push(self, priority, value) -> "IndexHeap":
        """Push item onto heap, maintaining the heap invariant."""
        n = len(self)
        heap = IndexHeap(
            self._priorities.at[n].set(priority), self._indices.at[n].set(value), n + 1
        )
        return heap._siftdown(0, n)

    def _less(self, i1: int, i2: int) -> bool:
        left = self._priorities[i1]
        right = self._priorities[i2]
        return lax.cond(
            left == right,
            lambda: self.indices[i1] < self.indices[i2],
            lambda: left < right,
        )

    def _siftup(self, pos: int):
        endpos = len(self)
        startpos = pos
        newitem = self[pos]
        childpos = 2 * pos + 1
        heap = self
        rightpos = 0
        s = pos, endpos, startpos, rightpos, newitem, childpos, heap
        T = tp.Tuple[int, int, int, int, HeapItem, int, IndexHeap]

        def cond_fn(s):
            pos, endpos, startpos, rightpos, newitem, childpos, heap = s
            return childpos < endpos

        def body_fn(s: T):
            pos, endpos, startpos, rightpos, newitem, childpos, heap = s
            del s
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            cond = jnp.logical_and(
                rightpos < endpos, jnp.logical_not(heap._less(childpos, rightpos))
            )
            childpos = lax.cond(cond, lambda: rightpos, lambda: childpos)

            # Move the smaller child up.
            heap = heap._replaceitem(pos, childpos)
            pos = childpos
            childpos = 2 * pos + 1
            return pos, endpos, startpos, rightpos, newitem, childpos, heap

        s: T = lax.while_loop(cond_fn, body_fn, s)
        pos, endpos, startpos, rightpos, newitem, childpos, heap = s
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        heap = heap._setitem(pos, newitem)
        heap = heap._siftdown(startpos, pos)
        return heap

    def _siftdown(self, startpos: int, pos: int) -> "IndexHeap":
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        newpr, newind = self[pos]
        parentpos = 0
        parentpr = jnp.zeros((), self._priorities.dtype)
        cont = True
        priorities = self._priorities
        indices = self._indices
        s = startpos, pos, newpr, newind, parentpos, parentpr, cont, priorities, indices

        def cond_fn(s):
            (
                startpos,
                pos,
                newpr,
                newind,
                parentpos,
                parentpr,
                cont,
                priorities,
                indices,
            ) = s
            return jnp.logical_and(pos > startpos, cont)

        def body_fn(s):
            (
                startpos,
                pos,
                newpr,
                newind,
                parentpos,
                parentpr,
                cont,
                priorities,
                indices,
            ) = s
            parentpos = (pos - 1) >> 1
            parentpr = self._priorities[parentpos]

            def if_true():
                return (
                    priorities.at[pos].set(parentpr),
                    indices.at[pos].set(indices[parentpos]),
                    parentpos,
                    True,
                )

            def if_false():
                return priorities, indices, pos, False

            priorities, indices, pos, cont = lax.cond(
                newpr < parentpr, if_true, if_false
            )
            return (
                startpos,
                pos,
                newpr,
                newind,
                parentpos,
                parentpr,
                cont,
                priorities,
                indices,
            )

        (
            startpos,
            pos,
            newpr,
            newind,
            parentpos,
            parentpr,
            cont,
            priorities,
            indices,
        ) = lax.while_loop(cond_fn, body_fn, s)

        return IndexHeap(priorities, indices, len(self))


@functools.partial(jax.jit, static_argnames="max_length")
def padded_index_heap(
    priorities: jnp.ndarray, indices: jnp.ndarray, max_length: int
) -> IndexHeap:
    length = priorities.size
    assert indices.size == length
    padding = max_length - length
    actual_priorities = jnp.pad(priorities, [[0, padding]])
    actual_indices = jnp.pad(indices, [[0, padding]])
    return IndexHeap(actual_priorities, actual_indices, length)
