#architecture

# Vision model

`DroneVisionExtractor` in `vision_model.py` (and duplicated inside
`train_vision.py` and `train_distill.py`). A coarse-to-fine CNN with
parallel state MLP, producing 256 features for SB3's MLP policy head.

## Architecture, actually shipped

This is the topology in the trained `policy.pth`, confirmed by
inspecting tensor shapes. SB3 stores three copies of the features
extractor (`features_extractor`, `pi_features_extractor`,
`vf_features_extractor`) — they are identical in our case; see note
below.

```
image (batch, 14, 48, 48) uint8 → /255 float
  │
  ├─► coarse_cnn
  │     Conv2d(14 → 16, k=5, s=2, p=1)  ReLU
  │     Conv2d(16 → 32, k=3, s=2)       ReLU
  │     → (batch, 32, 11, 11)          ← flatten 3872
  │
  ├─► fine_cnn  (takes the 32-channel coarse maps)
  │     Conv2d(32 → 48, k=3, s=1, p=1)  ReLU
  │     Conv2d(48 → 64, k=3, s=1)       ReLU
  │     → (batch, 64, 9, 9)            ← flatten 5184
  │
  │     coarse_fc: Linear(3872 → 64)  ReLU    ← 64D coarse features
  │     fine_fc:   Linear(5184 → 128) ReLU    ← 128D fine features
  │
state (batch, 19)
  │
  └─► state_mlp
        Linear(19 → 64)  ReLU
        Linear(64 → 64)  ReLU                 ← 64D state features

concat(coarse_64, fine_128, state_64) = 256D features
```

Feeds into SB3's `MlpExtractor`:

```
mlp_extractor.policy_net: Linear(256 → 128) ReLU → Linear(128 → 64) ReLU
                                                   → action_net: Linear(64 → 4)

mlp_extractor.value_net:  Linear(256 → 128) ReLU → Linear(128 → 64) ReLU
                                                   → value_net:  Linear(64 → 1)
```

Total parameters in the policy (from checkpoint): features extractor
replicated 3× (pi/vf/shared) + `MlpExtractor` policy/value pair +
action/value heads. The `.zip` is 13 MB including optimizer state.

## Why coarse-to-fine

Pattern from `event-sharp-nerf-drones/networks/pdrf/voxnerf.py` (hence
the "VoxelNeRF-inspired" comments in the code). The idea: the fine
stage **reuses the coarse feature maps** rather than processing the
raw image twice. Two advantages:

1. Faster forward pass than a single large CNN doing the same work,
   because the fine stage sees 32-channel 11×11 maps instead of
   14-channel 48×48 images.
2. Implicit multi-scale supervision — the coarse features must carry
   enough signal to support the fine stage, which encourages the
   coarse stage to learn robust mid-level features.

Whether this actually outperforms a single CNN of comparable capacity
has not been ablated in this project. Treat it as a design choice
inherited from the reference paper, not as a measured win.

## Why 6D rotation in the state branch

The 19D state vector leads with 6 values: the first two columns of the
body-to-world rotation matrix (Gram-Schmidt orthonormalized). Zhou et
al. (CVPR 2019, "On the Continuity of Rotation Representations in
Neural Networks") showed that Euler angles and unit quaternions are
**discontinuous** representations, which can hurt gradient flow when
the network is asked to regress against rotation. A 6D representation
via two columns is continuous and avoids gimbal lock.

For a CNN-adjacent policy that must reason about attitude while also
predicting rate commands, this choice matters. It's a decision worth
understanding rather than copying blindly — if you were rebuilding the
policy, you should still use 6D unless you had a specific reason
otherwise.

See [[decisions-log#6D rotation over quaternions]].

## SB3's three copies of the features extractor

The checkpoint has `features_extractor.*`, `pi_features_extractor.*`,
and `vf_features_extractor.*` — three identical CNN+MLP trees. This
is an SB3 2.x idiom: by default, policy and value share the same
extractor, but SB3 still saves a "shared" copy and per-head copies for
safety across configuration changes. For inference we can load any one
(and the deployment adapter loads only `features_extractor.*`). During
training the three were tied.

## The quirk the audit flagged — and the answer

The `dcl_adapter.py` `PolicyNet` is built as `Linear(275, 128)` with
input `cat(features_256, state_19)`. That's a fundamental mismatch
with the real architecture, where the state MLP's output is *already*
inside the 256D features and `mlp_extractor.policy_net` starts from
`(256, 128)`.

Reading the adapter source in isolation, it could look like a
deliberate design ("inject state twice for emphasis"). The checkpoint
says no: there is exactly one policy path, `Linear(256, 128)`, with no
state re-concatenation.

**The adapter was written wrong.** The `pi_net.` key prefix matches
nothing. `strict=False` hides this. The policy runs on random init.

See [[fragilities#The policy head is untrained at deployment]] for how to confirm,
and [[fragilities]] (same entry) for the replacement code.

## Input image format note

The CNN expects channels-first `(C, H, W)`. `DroneRaceEnv` returns
observations channels-last `(H, W, C)` inside the Dict space. SB3's
default `MultiInputPolicy` handles the transpose internally for image
spaces. But `dcl_adapter.process_observation` also transposes manually
to `(C, H, W)` before feeding the extractor — which works because the
feature extractor, when called outside SB3, takes whatever we give it.
Both code paths produce correct inputs; the training path goes through
SB3's wrappers, the deployment path bypasses them.

## See also

- [[perception]] — the sensors that produce the 14-channel image and 19D state
- [[training]] — how this extractor was trained (distillation from state expert)
- [[fragilities]] — the deployment wiring that breaks the policy head
- [[decisions-log]] — 6D rotation, coarse-to-fine CNN
