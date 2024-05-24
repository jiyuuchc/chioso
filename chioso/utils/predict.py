import logging
from functools import partial, lru_cache

import numpy as np
import jax
import pydensecrf.densecrf as dcrf

from xtrain import GeneratorAdapter, JIT

_cached_partial = lru_cache(partial)

def crf(U, *, sxy=20, compat=1.0, n_iter=20):
    U = np.array(U)
    if U.ndim != 3:
        raise ValueError(f"Only works on 2D data. Got shape {U.shape}")
    if U.shape[-1] == 1:
        U = np.concatenate([U, np.zeros_like(U)], axis=-1)

    h, w, c = U.shape

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(np.ascontiguousarray(U.transpose(2,0,1).reshape(c, -1)))

    d.addPairwiseGaussian(sxy=sxy, compat=compat)

    q = np.argmax(d.inference(n_iter), axis=0).reshape(h, w)

    return q


def crf_scan(U, *, ps=512, gs=500, sxy=1.0, compat=20, n_iter=5):
    if U.ndim != 3:
        raise ValueError(f"Only works on 2D data. Got shape {U.shape}")

    h, w, c = U.shape
    bs = (ps - gs) // 2

    output = np.zeros([h, w], int)

    for y0 in range(0, h, gs):
        for x0 in range(0, w, gs):
            v = crf(U[y0:y0+ps, x0:x0+ps], sxy=sxy, compat=compat, n_iter=n_iter)
            output[y0+bs : y0+bs+gs, x0+bs : x0+bs+gs] = v[bs:bs+gs,bs:bs+gs]

    return output
    

def get_seg(cts, score, *, score_threshold=0.1, offset=0.5, sxy=3.0, compat=5):
    from scipy.optimize import minimize_scalar
    score = score > score_threshold
    cts -= cts.mean()

    def _fun(d):
        u = cts + d
        q = u > 0
        its = np.count_nonzero(score & q)
        union = np.count_nonzero(score | q)
        return - its/union
    
    r = minimize_scalar(_fun, bounds=(-3,3))
    d = r.x
    u = cts + d - offset
    q = crf(u[..., None], sxy=sxy, compat=compat)

    return q
    
def predict(model, params, sgdataset, *, ps=1024, gs=1000, binning=4, bucketsize=524288, sxy=1.5, compat=20, n_iter=5, return_logits=False):
    h, w = sgdataset.shape

    if sxy < 0 or compat < 0 or n_iter < 0:
        raise ValueError("Invalid CRF hyperparameter. Negative value is not allowed")

    bs = (ps - gs) // binning // 2

    imgh = ((h - 1) // gs * gs + ps) // binning
    imgw = ((w - 1) // gs * gs + ps) // binning

    def _format_patch(sgc):
        if sgc.shape[0] != ps or sgc.shape[1] != ps:
            sgc = sgc.pad([[0, ps-sgc.shape[0]],[0, ps-sgc.shape[1]]])

        if binning != 1:
            sgc = sgc.binning([binning, binning])

        sgc = sgc.pad_to_bucket_size(bucket_size=bucketsize)

        return sgc

    def _gen(sgdataset):
        h, w = sgdataset.shape
        for y0 in range(0, h, gs):
            for x0 in range(0, w, gs):
                sgc = _format_patch(sgdataset[y0:y0+ps, x0:x0+ps])
                yield sgc, y0//binning, x0//binning


    def _method(mdl, sgc, y0, x0):
        pred = mdl(sgc, training=False)
        return pred, sgc, y0, x0

    apply_fn = _cached_partial(
        model.apply,
        mutable="adversal",
        method=_method,
    )

    logits = None
    label = np.zeros([imgh, imgw], dtype="uint32") 
    cts = np.zeros([imgh, imgw], dtype="uint32")
    scores = np.zeros([imgh, imgw], dtype="float32")

    for inputs in GeneratorAdapter(partial(_gen, sgdataset)):
        (pred, sg, y0, x0), y = JIT.predict(apply_fn, dict(params=params), inputs)

        if sg.indptr[-1] > 0:
            cts_ = sg.render(mode="counts")
            logits_ = np.array(pred["output"])
            scores_= np.array(pred["dsc_loss_main"])

            if sxy > 0 and n_iter > 0:
                e_cls = -jax.nn.log_softmax(logits_)
                label_ = crf(e_cls, sxy=sxy, compat=compat, n_iter=n_iter)
            else:
                label_ = np.argmax(logits_, axis=-1)

            y0, x0 = y0 + bs, x0 + bs
            y1, x1 = y0 + gs//binning, x0 + gs//binning

            label[y0 : y1, x0 : x1] = label_[bs:-bs,bs:-bs]
            cts[y0 : y1, x0 : x1] = cts_[bs:-bs,bs:-bs]
            scores[y0 : y1, x0 : x1] = scores_[bs:-bs,bs:-bs, 0]

        if return_logits:
            if logits is None:
                n_cls = pred["output"].shape[-1]
                logits = np.zeros([imgh, imgw, n_cls], dtype="float32")
            logits[y0 : y1, x0 : x1] = logits_[bs:-bs,bs:-bs]

    if return_logits:
        return label, scores, cts, logits
    else:
        return label, scores, cts

