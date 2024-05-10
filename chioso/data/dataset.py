import logging

from functools import lru_cache

import dataclasses
import h5py
import numpy as np

from .sgdata import SGData2D

def _format_slice(x, m):
    if isinstance(x, int):
        x = slice(x, x + 1)
    start, stop, step = x.indices(m)
    if step > 0:
        stop = max(stop, start)
    else:
        stop = min(stop, start)
    return start, stop, step

@dataclasses.dataclass
class SGDataset2D:
    group: h5py.Group
    dtype: np.dtype = np.dtype(int)

    @property
    def shape(self):
        return self.group.attrs["shape"]
        
    @property
    def n_genes(self):
        return self.group.attrs["n_genes"]

    @property
    def block_size(self):
        return self.group.attrs["block_size"]

    def __post_init__(self):
        if self.group is not None:
            if self.group.attrs["type"] == "sgdataset2d":
                self._read_block = lru_cache(maxsize=8)(self.__read_block)
                return

        raise ValueError(f"The underlying h5 group {self.group} is not a proper SGDataset")

    def __repr__(self):
        if self.group is None:
            return "Closed SGDataset"
        else:
            return f"SGDataset of 2d shape {self.shape} and {self.n_genes} genes."

    def _empty_block(self):
        sy, sx = self.block_size

        return SGData2D(
            np.array([], dtype=self.dtype),
            np.array([], dtype=int),
            np.zeros([sy*sx+1], dtype=int),
            (sy, sx),
            self.n_genes,
        )

    def _write_block(self, sg, block_loc):
        y0, x0 = block_loc
        sy, sx = self.block_size
        h0, w0 = sg.shape

        assert y0 % sy == 0
        assert x0 % sx == 0

        blockname = f"block_{y0}_{x0}"
        self.group.create_group(blockname)

        sg = sg.pad([[0, sy-sg.shape[0]], [0, sx -sg.shape[1]]])
        assert sg.shape[0] == sy
        assert sg.shape[1] == sx

        self.group[blockname]["data"] = sg.data
        self.group[blockname]["indices"] = sg.indices
        self.group[blockname]["indptr"] = sg.indptr

        h, w = self.shape
        h = max(h, y0 + h0)
        w = max(w, x0 + w0)
        self.group.attrs["shape"] = h, w
    
    def __read_block(self, block_loc):
        y0, x0 = block_loc
        sy, sx = self.block_size

        assert y0 % sy == 0
        assert x0 % sx == 0

        if y0 >= self.shape[0] or x0 >= self.shape[1]:
            return ValueError(f"block {y0}-{x0} is out-of-bound")
    
        blockname = f"block_{y0}_{x0}"
        if not blockname in self.group:
            return self._empty_block()

        return SGData2D(
            self.group[blockname]["data"][...],
            self.group[blockname]["indices"][...],
            self.group[blockname]["indptr"][...],
            (sy, sx),
            self.n_genes,
        )
    
    def _get_2d_range(self, bbox):
        logging.debug(f"retrieving range {bbox}")

        y0, x0, y1, x1 = bbox
        sy, sx = self.block_size
        y0b = y0 // sy * sy
        x0b = x0 // sx * sx
        y1b = (y1-1) // sy * sy
        x1b = (x1-1) // sx * sx

        logging.debug(f"retrieving blocks x: {x0b} - {x1b}, y:{y0b} - {y1b}")

        sg_list = []
        for xb in range(x0b, x1b + sx, sx):
            sg_list_inner = []
            for yb in range(y0b, y1b + sy, sy):
                sg_list_inner.append(self._read_block((yb, xb)))
            sg_list.append(SGData2D.vstack(sg_list_inner))
        sg = SGData2D.hstack(sg_list)

        return sg, y0b, x0b
    
    def __getitem__(self, slices):
        try:
            slicey, slicex = slices
        except:
            raise ValueError(f"SGDataset2D supports 2D slice indexing only. Got {slices}")

        starty, stopy, stepy = _format_slice(slicey, self.shape[0])
        startx, stopx, stepx = _format_slice(slicex, self.shape[1])

        if starty == stopy or startx == stopx :
            return SGData2D(
                np.array([], dtype=self.dtype),
                np.array([], dtype=int),
                np.array([], dtype=int),
                (0,0),
                self.n_genes,
            )

        if stepy > 0:
            miny, maxy = starty, stopy
        else:
            miny, maxy = stopy+1, starty+1

        if stepx > 0:
            minx, maxx = startx, stopx
        else:
            minx, maxx = stopx+1, startx+1
        
        sg, block_y0, block_x0 = self._get_2d_range((miny, minx, maxy, maxx))
        slicey = slice(starty - block_y0, stopy - block_y0, stepy)
        slicex = slice(startx - block_x0, stopx - block_x0, stepx)

        return sg.__getitem__((slicey, slicex))

    @classmethod
    def create(cls, parent, name, *, block_size=(1024,1024), n_genes=-1, dtype=int):
        grp = parent.create_group(name)
        grp.attrs["type"]="sgdataset2d"
        grp.attrs["ver"]="0.0"
        grp.attrs["shape"]=(0,0)
        grp.attrs["block_size"]=block_size
        grp.attrs["n_genes"] = n_genes
        return cls(grp, np.dtype(dtype))

    @classmethod
    def create_from_sgdata(cls, parent, name, sg, *, block_size=(1024, 1024)):
        obj = cls.create(parent, name, block_size=block_size, n_genes=sg.n_genes, dtype=sg.dtype)
        h, w = sg.shape
        sy, sx = block_size

        for y0 in range(0, h, sy):
            for x0 in range(0, w, sx):
                sgc = sg[y0:min(y0+sy, h), x0:min(x0+sx, w)]
                obj._write_block(sgc, (y0, x0))
        
        return obj
