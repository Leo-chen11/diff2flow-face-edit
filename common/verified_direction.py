"""Which way a training edit goes, and when the model's own reading may be overruled.

The flow edits FROM its conditioner's reading of the source. Training normally takes the edit's
direction from that reading. This module decides, per edit, one of three things:

  follow   the conditioner's reading decides (nothing contradicts it clearly). Almost every edit.
  correct  the conditioner is SHOWN wrong: every reading of every independent view of the face is
           clear and on one side, the conditioner is on the other side (or within the margin of 0.5),
           and the weakest of those readings is at least as confident as the conditioner (+ beat).
           The edit then takes that direction and the reading rule of inference (0.5 -/+ margin).
  abstain  the evidence contradicts the conditioner but does not prove it wrong: some view reads the
           face clearly on the other side, but not all of them, or they are less sure than the model.
           Nobody can tell who is right, so the edit gets no direction supervision at all (the
           target / CLIP / DDS terms skip it; identity, preservation and the flow's likelihood stay).

"Views" are readings of the SAME face that fail in different ways: two teachers of different
architecture (r34, ConvNeXt), each on the photo, its mirror image and the e4e reconstruction (what
the flow actually edits: an attribute the inversion lost is not there to edit), plus the r34 reading
precomputed with the data. A correction needs ALL of them to agree, so one network fooled by one
view (a pose, a reflection, an inversion artefact) cannot trigger it. The evaluation judges (R50,
CelebA r18) are never views.

scripts/audit_verified_direction.py applies this same function to the training set and writes the
corrected and abstained faces for a human to check: the rate at which a correction overrules a
conditioner that was in fact right is measured, not assumed.
"""
import torch


def verified_direction(cond, reads, margin=0.1, clear=0.65, beat=0.0):
    """cond: (B,) the conditioner's reading of the edited attribute (0..1).
    reads: (V, B) every view's reading of it (0..1).
    margin: the reading rule's margin; clear: a view is clear above it or below 1 - clear;
    beat: the weakest view must be at least this much MORE confident (|p - 0.5|) than the
    conditioner when the conditioner is clearly on the other side.

    Returns a dict of (B,) tensors: rm (the edit's direction, True = remove), correct, abstain,
    verified (all views clear and on one side), reading (the conditioner reading to encode and
    target with: moved by the reading rule where correct, unchanged elsewhere)."""
    reads = torch.as_tensor(reads, device=cond.device, dtype=cond.dtype)
    hi = (reads > clear).all(0)
    lo = (reads < 1.0 - clear).all(0)
    verified = hi | lo
    other_side = torch.where(hi, cond < 0.5, cond > 0.5) & verified
    near = torch.where(hi, cond < 0.5 + margin, cond > 0.5 - margin) & verified
    view_conf = (reads - 0.5).abs().min(0).values
    cond_conf = (cond - 0.5).abs()
    strong = view_conf >= cond_conf + beat
    correct = near & (~other_side | strong)
    # some view reads the face clearly on the side the conditioner does NOT
    contradicted = torch.where(cond > 0.5, (reads < 1.0 - clear).any(0), (reads > clear).any(0))
    abstain = contradicted & ~correct
    rm = torch.where(correct, hi, cond > 0.5)
    moved = torch.where(rm, cond.clamp(min=0.5 + margin), cond.clamp(max=0.5 - margin))
    reading = torch.where(correct, moved, cond)
    return {'rm': rm, 'correct': correct, 'abstain': abstain, 'verified': verified, 'reading': reading}
