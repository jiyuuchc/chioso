from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.typing import ArrayLike
from scipy.sparse import coo_array, csr_array


class SGData2D(struct.PyTreeNode):
    """Immutable object representing 2D SG data, suitable as input for JAX function.
    Internally it is similar to csr_array

    Attributes:
        data: all mRNA cnts, should be integer. The index range of of i-th pixel is [indptr[i], indptr[i+1]).
            might be padded with 0
        indices: the gene_idx of each element in data. Same length as data. might be padded with -1
        indptr: index array of compressed sparse row representation. len(indptr) = height x width + 1.
        shape: tuple of (height, width)
        n_genes: number of genes
    """

    data: ArrayLike
    indices: ArrayLike
    indptr: ArrayLike

    shape: tuple[int, int] = struct.field(pytree_node=False)
    n_genes: int = struct.field(pytree_node=False)
    bucket_size: int = struct.field(pytree_node=False, default=-1)

    def to_csr(self):
        n_valid = self._get_n_valid()

        data = self.data[:n_valid]
        indices = self.indices[:n_valid]
        indptr = self.indptr

        n_pixels = self.shape[0] * self.shape[1]

        assert len(indptr) == n_pixels + 1
        assert indptr[-1] == len(data)

        return csr_array((data, indices, indptr), (n_pixels, self.n_genes))

    def _get_n_valid(self):
        return self.indptr[-1]

    def _get_segms(self, use_jax=False):
        if use_jax:
            rpt = jnp.diff(self.indptr)  # h x w
            segms = jnp.repeat(
                jnp.arange(len(rpt)),
                rpt,
                total_repeat_length=self.data.shape[0],
            )
        else:
            rpt = np.diff(self.indptr)  # h x w
            segms = np.repeat(
                np.arange(len(rpt)),
                rpt,
            )
            segms = np.pad(
                segms, [0, self.data.shape[0] - len(segms)], constant_values=segms[-1]
            )

        assert len(segms) == self.data.shape[0]

        return segms

    def render(self, mode: str = "gene", use_jax=False) -> np.ndarray:
        """Render a 2D image representing the SGData

        Args:
            mode: "gene"|"counts"|"max"

        Returns:
            rendered image.

        """
        np_ = jnp if use_jax else np

        rendered = np_.zeros(
            (self.shape[0] * self.shape[1],) + self.data.shape[1:],
            dtype=self.data.dtype,
        )

        segms = np_.clip(
            self._get_segms(use_jax=use_jax), 0, len(rendered) - 1
        )  # avoid indexing out-of-bound error

        if mode == "gene" or mode == "counts":

            data = self.data if mode == "counts" else (self.data > 0).astype("int32")
            np_.add.at(rendered, segms, data)

        elif mode == "max":

            np_.maximum.at(rendered, segms, self.data)

        else:
            raise ValueError(f"Unknown op mode: {mode}")

        rendered = rendered.reshape(self.shape + self.data.shape[1:])

        return rendered

    def __getitem__(self, items):
        csr = self.to_csr()

        def format_slice(x, m):
            x0 = x
            # m = int(m)
            if isinstance(x, int):
                x = slice(x, x + 1)
            if x.stop is not None and x.stop < 0:
                x = slice(x.start, x.stop + m, x.step)
            if x.start is not None and x.start < 0:
                x = slice(x.start + m, x.stop, x.step)
            if x.step is None:
                x = slice(x.start, x.stop, 1)
            if x.stop is None:
                real_stop = -1 if x.step < 0 else m
                x = slice(x.start, real_stop, x.step)
            if x.start is None:
                real_start = m - 1 if x.step < 0 else 0
                x = slice(real_start, x.stop, x.step)

            # if not (
            #     isinstance(x.start, int) and
            #     isinstance(x.stop, int) and
            #     isinstance(x.step, int)
            # ):
            #     raise ValueError(f"Invalid slice expression ({x0.start}, {x0.stop}, {x0.step})")

            return x

        items = [format_slice(s, m) for s, m in zip(items, self.shape)]

        idx = np.mgrid[items]

        crop = csr[np.ravel_multi_index(idx, self.shape, mode="clip").flat]

        return self.from_csr(crop, idx[0].shape, self.bucket_size)

    def pad_to_bucket_size(self, bucket_size: int) -> SGData2D:
        if bucket_size <= 0:
            raise ValueError(f"bucket_size should be positive, got {bucket_size}")

        # remove existing padding
        if len(self.data) > 0:
            n_valid = self._get_n_valid()
            data = self.data[:n_valid]
            indices = self.indices[:n_valid]

            # new padding
            padding = ((len(data) - 1) // bucket_size + 1) * bucket_size - len(data)
        else:
            data = self.data
            indices = self.indices
            padding = bucket_size

        return self.replace(
            data=np.pad(data, [0, padding], constant_values=0),
            indices=np.pad(indices, [0, padding], constant_values=-1),
            bucket_size=bucket_size,
        )

    def region_counts(self, label):
        from skimage.measure import regionprops

        if label.shape != self.shape:
            raise ValueError(
                f"label has the shape: {label.shape}, which is different from that of the SGData {self.shape}"
            )

        max_idx = label.max()

        csr = self.to_csr()

        cnts = np.zeros([max_idx, self.n_genes], dtype=self.data.dtype)

        for rp in regionprops(label):
            idx = rp["label"]
            coords = rp["coords"]
            coords = coords[:, 0] * self.shape[1] + coords[:, 1]
            cnts[idx - 1] = csr[coords, :].sum(axis=0)

        return cnts

    def binning(self, bin_size: tuple[int, int], *, prune=True) -> SGData2D:
        inner_idx = np.mgrid[: bin_size[0], : bin_size[1]]
        h, w = self.shape
        hh = h // bin_size[0] * bin_size[0]
        ww = w // bin_size[1] * bin_size[1]
        outer_idx = np.mgrid[: hh : bin_size[0], : ww : bin_size[1]]

        idx = outer_idx.reshape(2, -1, 1) + inner_idx.reshape(2, 1, -1)
        idx = idx[0] * w + idx[1]

        csr = self.to_csr()
        csr = csr[idx.flat]
        csr = csr_array(
            (csr.data, csr.indices, csr.indptr[:: bin_size[0] * bin_size[1]]),
            (h * w // bin_size[0] // bin_size[1], self.n_genes),
        )
        if prune:
            coo = csr.tocoo()
            csr = csr_array((coo.data, (coo.row, coo.col)), csr.shape)

        return self.from_csr(csr, outer_idx.shape[1:], bucket_size=self.bucket_size)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        idx = np.ravel_multi_index(
            np.mgrid[0 : self.shape[0], 0 : self.shape[1]],
            self.shape,
        ).T.reshape(-1)

        return self.from_csr(
            self.to_csr()[idx], (self.shape[1], self.shape[0]), self.bucket_size
        )

    def rearrange_gene_indices(self, lut, *, n_genes=None):
        indices = lut[self.indices]
        valid = indices >= 0
        indices = np.where(valid, indices, 0)
        cnts = np.where(valid, self.data,  0)

        if n_genes is None:
            n_genes = indices.max() + 1

        return self.replace(
            data=cnts,
            indices=indices,
            n_genes = n_genes,
        )


    @classmethod
    def from_csr(cls, csr: csr_array, shape: tuple[int, int], bucket_size: int = -1):
        csr.eliminate_zeros()
        obj = cls(
            data=csr.data,
            indices=csr.indices,
            indptr=csr.indptr,
            shape=shape,
            n_genes=csr.shape[1],
        )

        if bucket_size > 0:
            obj = obj.pad_to_bucket_size(bucket_size=bucket_size)

        return obj

    @classmethod
    def from_h5ad(
        cls,
        h5adfile: str | Path,
        lut: Callable | Mapping | ArrayLike | None = None,
        *,
        n_genes: int | None = None,
        return_genes: bool = False,
        bucket_size: int = -1,
    ):
        import h5py

        with h5py.File(h5adfile, mode="r") as f:
            cnts = np.asarray(f["X"]["data"])
            indices = np.asarray(f["X"]["indices"])
            indptr = np.asarray(f["X"]["indptr"])
            genes = list(f["var/index"])
            h, w = f["X"].attrs["2D_dimension"]

        if lut is not None:
            if isinstance(lut, Callable):
                lut_arr = np.asarray([lut(g) for g in genes]).astype(int)
            elif isinstance(lut, np.ndarray) or isinstance(lut, jnp.ndarray):
                lut_arr = np.asarray(lut)
            else:
                lut_arr = np.asarray(
                    [lut[g] if g in lut else -1 for g in genes]
                ).astype(int)
            indices = lut_arr[indices]
            valid = indices >= 0
            indices = np.where(valid, indices, 0)
            cnts[~valid] = 0

            if n_genes is None:
                n_genes = indices.max() + 1

        if n_genes is None:
            n_genes = len(genes)

        csr = csr_array((cnts, indices, indptr), shape=(h * w, n_genes))
        obj = cls.from_csr(csr, (h, w), bucket_size=bucket_size)

        if return_genes:
            return obj, genes
        else:
            return obj
