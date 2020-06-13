import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import ctc
import chainer_ctc as c_ctc


T, B, C = 128, 256, 32
t = T // 2 - 4
blank = 0
device = 'cuda'
seed = 1
atol = 1e-3
for set_seed in [torch.manual_seed] + ([torch.cuda.manual_seed_all] if device == 'cuda' else []):
    set_seed(seed)


def tictoc():
    return (device == 'cuda' and torch.cuda.synchronize()) or time.time()


logits = torch.randn(T, B, C, device=device).requires_grad_()
targets = torch.randint(blank + 1, C, (B, t), dtype=torch.long, device=device)
input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
target_lengths = torch.full((B,), t, dtype=torch.long, device=device)
log_probs = logits.log_softmax(dim=-1)

print('Device:', device)
print('Log-probs shape (time X batch X channels):',
      'x'.join(map(str, log_probs.shape)))

tic = tictoc()
builtin_ctc = F.ctc_loss(log_probs, targets, input_lengths,
                         target_lengths, blank=0, reduction='none')
toc = tictoc()
builtin_ctc_grad, = torch.autograd.grad(
    builtin_ctc.sum(), logits, retain_graph=True)
print('Built-in CTC loss', 'fwd', toc - tic, 'bwd', tictoc() - toc)

tic = tictoc()
custom_ctc = ctc.ctc_loss(
    log_probs, targets, input_lengths, target_lengths, blank=0, reduction='none')
toc = tictoc()
custom_ctc_grad, = torch.autograd.grad(
    custom_ctc.sum(), logits, retain_graph=True)
print('Custom CTC loss', 'fwd', toc - tic, 'bwd', tictoc() - toc)

tic = tictoc()
chainer_ctc = c_ctc.ctc_loss(
    log_probs, targets, input_lengths, target_lengths, blank=0, reduction='none')
toc = tictoc()
chainer_ctc_grad, = torch.autograd.grad(
    chainer_ctc.sum(), logits, retain_graph=True)
print('Chainer CTC loss', 'fwd', toc - tic, 'bwd', tictoc() - toc)

ce_alignment_targets = ctc.ctc_alignment_targets(
    log_probs, targets, input_lengths, target_lengths, blank=0)
ce_ctc = -ce_alignment_targets * log_probs
ce_ctc_grad, = torch.autograd.grad(ce_ctc.sum(), logits, retain_graph=True)

print('Custom loss matches:', torch.allclose(
    builtin_ctc, custom_ctc, atol=atol))
print('Grad matches:', torch.allclose(
    builtin_ctc_grad, custom_ctc_grad, atol=atol))
print('CE grad matches:', torch.allclose(
    builtin_ctc_grad, ce_ctc_grad, atol=atol))

alignment = ctc.ctc_loss(log_probs, targets, input_lengths,
                         target_lengths, blank=0, reduction='none', alignment=True)
a = alignment[:, 0, :target_lengths[0]]

fig, ax = plt.subplots(2, 1)

ax[0].set_title('Input-Output Viterbi alignment')
ax[0].imshow(a.t().cpu(), origin='lower', aspect='auto')
ax[0].set_xlabel('Input steps')
ax[0].set_ylabel('Output steps')

ax[1].set_title('CTC alignment targets')
a = ce_alignment_targets[:, 0, :]
ax[1].imshow(a.t().cpu(), origin='lower', aspect='auto')
ax[1].set_xlabel('Input steps')
ax[1].set_ylabel(f'Output symbols, blank {blank}')

plt.subplots_adjust(hspace=0.5)
plt.savefig('alignment.png')
