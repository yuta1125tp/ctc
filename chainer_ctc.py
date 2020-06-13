# just adjust chainer's ctc to torch.
# [chainer/functions/loss/ctc.py](https://github.com/chainer/chainer/blob/v7.4.0/chainer/functions/loss/ctc.py)

# pylint: disable=E1101 # ignore "Module '' has no '' member "


import numpy
import torch
from torch import functional as F
from torch import nn


def get_array_module(arr):
    if isinstance(arr, torch.Tensor):
        return torch
    else:
        return backend.get_array_module(arr)


def int_dtype(xp):
    return torch.long if xp is torch else xp.int32


def inv_slice(length, **kwargs):
    return torch.arange(length-1, -1, -1, dtype=torch.long, **kwargs)


def _logsumexp(a, xp, axis=None):
    if xp is torch:
        vmax, _vidx = xp.max(a, dim=axis, keepdims=True)
    else:
        vmax = xp.amax(a, axis=axis, keepdims=True)
    if xp is numpy or xp is torch:
        vmax += xp.log(xp.sum(xp.exp(a - vmax),
                              axis=axis, keepdims=True, dtype=a.dtype))
    else:
        _logsumexp_impl = cuda.reduce(
            'T x, T vmax', 'T y',
            'exp(x - vmax)', 'a + b', 'y += log(a)', '0',
            'logsumexp_impl')
        _logsumexp_impl(a, vmax, vmax, axis=axis, keepdims=True)
    return xp.squeeze(vmax, axis=axis)


def _label_to_path(labels, blank_symbol, xp):
    path = xp.full((len(labels), labels.shape[1] * 2 + 1),
                   blank_symbol, dtype=int_dtype(xp))
    path[:, 1::2] = labels
    return path


def _flip_path(path, path_length, xp):
    """Flips label sequence.

    This function rotates a label sequence and flips it.
    ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[b, t] = path[b, t + path_length[b]]``

    .. ::

       a b c d .     . a b c d    d c b a .
       e f . . .  -> . . . e f -> f e . . .
       g h i j k     g h i j k    k j i h g

    """
    n_batch, n_label = path.shape
    rotate = (xp.arange(n_label) + path_length[:, None]) % n_label
    if xp is torch:
        _slice = inv_slice(n_label)
    else:
        _slice = slice(None, None, -1)
    return path[xp.arange(n_batch, dtype=int_dtype(xp))[:, None],
                rotate][:, _slice]


def _flip_label_probability(y, input_length, xp):
    """Flips a label probability matrix.

    This function rotates a label probability matrix and flips it.
    ``y[i, b, l]`` stores log probability of label ``l`` at ``i``-th
    input in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, b, l] = y[i + input_length[b], b, l]``

    """
    seq, n_batch, n_vocab = y.shape
    rotate = (xp.arange(seq, dtype=int_dtype(xp))
              [:, None] + input_length) % seq
    if xp is torch:
        _slice = inv_slice(seq)
    else:
        _slice = slice(None, None, -1)
    return y[
        rotate[:, :, None],
        xp.arange(n_batch, dtype=int_dtype(xp))[None, :, None],
        xp.arange(n_vocab, dtype=int_dtype(xp))[None, None, :]][_slice]


def _flip_path_probability(prob, input_length, path_length, xp):
    """Flips a path probability matrix.

    This function returns a path probability matrix and flips it.
    ``prob[i, b, t]`` stores log probability at ``i``-th input and
    at time ``t`` in a output sequence in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, j, k] = prob[i + input_length[j], j, k + path_length[j]]``

    """
    seq, n_batch, n_label = prob.shape
    rotate_input = ((xp.arange(seq, dtype=int_dtype(xp))[:, None] + input_length)
                    % seq)
    rotate_label = ((xp.arange(n_label, dtype=int_dtype(xp)) + path_length[:, None])
                    % n_label)
    if xp is torch:
        _slice0 = inv_slice(seq)
        _slice2 = inv_slice(n_label)
        return prob[
            rotate_input[:, :, None],
            xp.arange(n_batch, dtype=int_dtype(xp))[None, :, None],
            rotate_label][_slice0][:, :, _slice2]
    else:
        return prob[
            rotate_input[:, :, None],
            xp.arange(n_batch, dtype=int_dtype(xp))[None, :, None],
            rotate_label][::-1, :, ::-1]


# path probability to label probability
def label_probability(label_size, path, path_length,
                      multiply_seq, xp):
    seq_length = len(multiply_seq)
    n_batch = len(path)
    dtype = multiply_seq.dtype

    ret = xp.zeros((seq_length, n_batch, label_size), dtype=dtype)
    if xp is numpy or xp is torch:
        for b in range(len(path)):
            target_path = path[b, :path_length[b]]
            chars = {c for c in target_path}
            for c in chars:
                ret[:, b, c] = xp.sum(
                    multiply_seq[:, b, 0:path_length[b]]
                    [:, target_path == c], axis=1)
    else:
        utils.nondeterministic('atomicAdd')
        cuda.elementwise(
            'T prob, I path, I path_length, I max_path_length',
            'raw T cum_prob',
            '''
            I t = i % max_path_length;
            if (t < path_length) {
                int n_batch = cum_prob.shape()[1];
                I s = i / (max_path_length * n_batch);
                I b = (i - s * (max_path_length * n_batch))
                    / max_path_length;
                int ind[] = {s, b, path};
                atomicAdd(&cum_prob[ind], prob);
            }
            ''', 'ctc_label_prob_sum'
        )(multiply_seq, path, path_length[:, None], path.shape[1], ret)
    return ret


def _computes_transition(
        prev_prob, path, path_length, cum_prob, y, zero_padding):
    xp = get_array_module(prev_prob)

    if xp is numpy or xp is torch:
        n_batch, max_path_length = path.shape
        mat = xp.full(
            (3, n_batch, max_path_length), zero_padding, dtype=y.dtype)
        mat[0, :, :] = prev_prob
        mat[1, :, 1:] = prev_prob[:, :-1]
        mat[2, :, 2:] = prev_prob[:, :-2]
        # disable transition between the same symbols
        # (including blank-to-blank)
        same_transition = (path[:, :-2] == path[:, 2:])
        mat[2, :, 2:][same_transition] = zero_padding
        prob = _logsumexp(mat, xp, axis=0)
        outside = xp.arange(max_path_length) >= path_length[:, None]
        prob[outside] = zero_padding
        cum_prob += prob
        batch_index = xp.arange(n_batch, dtype=int_dtype(xp))
        prob += y[batch_index[:, None], path]
    else:
        prob = xp.empty_like(prev_prob)
        cuda.elementwise(
            'raw T prob, raw I path, I path_length, T zero, raw T y',
            'T z, T cum_prob',
            '''
            int length = prob.shape()[1];
            int b = i / length;
            int t = i - b * length;
            if (t >= path_length) {
                z = zero;
                cum_prob += zero;
                return;
            }
            int ind1[] = {b, t};
            int ind2[] = {b, t - 1};
            int ind3[] = {b, t - 2};
            T f1 = prob[ind1];
            T f2 = (0 <= t - 1) ? prob[ind2] : zero;
            T f3 = (0 <= t - 2 && path[ind3] != path[ind1]) ?
                prob[ind3] : zero;

            // calculates log-sum-exp
            T m = max(f1, max(f2, f3));
            z = m + log(exp(f1 - m) + exp(f2 - m) + exp(f3 - m));

            cum_prob += z;

            int y_ind[] = {b, path[ind1]};
            z += y[y_ind];
            ''', 'ctc_transition'
        )(prev_prob, path, path_length[:, None], zero_padding, y,
            prob, cum_prob)
    return prob


def calc_trans(yseq, input_length,
               label, label_length, path, path_length, zero_padding, xp):
    max_input_length, n_batch, n_unit = yseq.shape
    max_label_length = label.shape[1]
    max_path_length = path.shape[1]
    assert label.shape == (n_batch, max_label_length), label.shape
    assert path.shape == (n_batch, max_label_length * 2 + 1)

    forward_prob = xp.full(
        (n_batch, max_path_length), zero_padding, dtype=yseq.dtype)
    forward_prob[:, 0] = 0
    backward_prob = forward_prob

    batch_index = xp.arange(n_batch, dtype=int_dtype(xp))
    seq_index = xp.arange(len(yseq), dtype=int_dtype(xp))
    prob = yseq[seq_index[:, None, None], batch_index[:, None], path]
    # forward computation.
    for i, y in enumerate(yseq):
        forward_prob = _computes_transition(
            forward_prob, path, path_length, prob[i], y, zero_padding)

    r_path = _flip_path(path, path_length, xp)

    yseq_inv = _flip_label_probability(yseq, input_length, xp)
    prob = _flip_path_probability(prob, input_length, path_length, xp)

    for i, y_inv in enumerate(yseq_inv):
        backward_prob = _computes_transition(
            backward_prob, r_path, path_length, prob[i], y_inv, zero_padding)

    return _flip_path_probability(prob, input_length, path_length, xp)


class NaiveCTC(torch.autograd.function.Function):

    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank_symbol, reduction):
        ctx.reduction = reduction

        xp = get_array_module(log_probs)

        if log_probs.dtype == numpy.float16:
            zero_padding = -10000.0
        else:
            zero_padding = -10000000000.0

        path_lengths = 2 * target_lengths + 1

        # self.yseq = _softmax(xs, xp)
        # log_yseq = self.log_matrix(self.yseq, xp)
        path = _label_to_path(targets, blank_symbol, xp)
        prob_trans = calc_trans(
            log_probs, input_lengths, targets,
            target_lengths, path, path_lengths, zero_padding, xp)

        loss = -_logsumexp(prob_trans[0], xp, axis=1)
        if ctx.reduction == 'mean':
            loss = loss.mean()
        ctx.save_for_backward(log_probs, targets, input_lengths, path_lengths, path, prob_trans)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, path_lengths, path, prob_trans = ctx.saved_tensors

        xp = get_array_module(grad_output)
        batch_size = targets.size(0)

        total_probability = _logsumexp(prob_trans[0], xp, axis=1)
        label_prob = label_probability(
            log_probs.size(2), path, path_lengths,
            xp.exp(prob_trans - total_probability[:, None]), xp)
        yseq = log_probs.exp() - label_prob
        if ctx.reduction == 'mean':
            yseq *= grad_output[0] / batch_size
        else:
            yseq *= grad_output[0][..., None]
        # mask
        yseq *= (xp.arange(len(yseq))[:, None] < input_lengths)[..., None]

        return yseq, None, None, None, None, None


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank : int = 0, reduction : str = 'none'):
    """"""
    return NaiveCTC.apply(log_probs, targets, input_lengths, target_lengths, blank, reduction)
