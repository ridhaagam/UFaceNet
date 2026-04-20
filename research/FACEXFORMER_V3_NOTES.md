# FaceXFormer v3 Notes For UFaceNet

Source: https://arxiv.org/pdf/2403.12960v3

## Paper Facts To Preserve

FaceXFormer v3 presents an end-to-end unified transformer for ten face tasks:

- face parsing
- landmark detection
- head pose estimation
- attribute prediction
- age estimation
- gender estimation
- race estimation
- face visibility prediction
- facial expression recognition
- face recognition

The paper represents tasks as learnable tokens and uses a lightweight FaceX decoder with task self-attention, task-to-face cross-attention, and face-to-task cross-attention. The reported design uses multi-scale encoder features, MLP fusion, two FaceX decoder layers, and task-specific output heads.

Important reported numbers from the paper:

- FaceXFormer handles ten tasks in one model.
- The paper reports 33.21 FPS in FP32 and 100.1 FPS in FP16.
- The computational comparison reports 114 GFLOPs for FaceXFormer versus 167 GFLOPs for Faceptor.
- The paper reports 109.29M parameters for the Swin-B configuration.
- Reported training uses eight A6000 GPUs, 224 x 224 input, AdamW, total batch size 384, 12 epochs, initial learning rate 1e-4, weight decay 1e-5, and learning rate drops at epochs 6 and 10.

## Datasets And Metrics In The Paper

Training datasets:

- FP: CelebAMask-HQ
- LD: 300W
- HPE: 300W-LP
- Attr: CelebA
- Exp: RAF-DB and AffectNet
- Age/Gen/Race: UTKFace and FairFace
- FR: MS1MV3
- Vis: COFW

Test datasets:

- FP: CelebAMask-HQ
- LD: 300W and 300VW
- HPE: BIWI
- Attr: CelebA and LFWA
- Exp: RAF-DB
- Age/Gen/Race: UTKFace and FairFace
- FR: LFW, CFP-FP, AgeDB, CALFW, CPLFW
- Vis: COFW

Metrics:

- FP: F1-score
- LD: NME
- HPE: MAE
- Age: MAE
- Attr/Gen/Race/Exp: accuracy
- FR: 1:1 verification accuracy
- Vis: recall at 80 percent precision

## Code Independence Requirement

UFaceNet must not use the FaceXFormer repository, runtime code, or checkpoints. FaceXFormer is a cited paper baseline and task/metric reference only.

Implementation work must stay in `ufacenet/`, with UFaceNet trained from scratch using its own configs, dataloaders, losses, and evaluators. If a paper comparison uses FaceXFormer values, those values must be cited and recorded in the benchmark ledger rather than produced through the upstream codebase.

## UFaceNet Opportunity

The paper already frames face analysis as a set of learnable task tokens. UFaceNet should add a new task token family for reconstruction/generation:

- `T_frec`: face reconstruction/generation token
- optional `T_geom`: geometry token for mesh/depth/normal
- optional `T_tex`: texture/albedo token
- optional `T_render`: render consistency token

The novelty should not be "we added another MLP." The stronger idea is a reconstruction-consistency block that uses the reconstruction path to regularize and verify the shared face representation.

The FRec path should remain one-pass and token-conditioned: a single UFaceNet forward call should be able to return analysis outputs and high-fidelity reconstruction/generation outputs when the requested task set includes FRec.

## Paper Claim Boundary

Safe claim:

UFaceNet extends tokenized unified facial analysis with a reconstruction/generation output family and evaluates the extension with face-specific reconstruction, identity, geometry, and distribution metrics.

Claim that needs stronger evidence:

UFaceNet is the first unified model to combine all ten FaceXFormer tasks with face reconstruction/generation in one real-time transformer.

Do not use the stronger claim until the related-work table includes 3D reconstruction and face generation baselines.
