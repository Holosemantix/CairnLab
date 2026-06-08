# Reference Audit

Date: 2026-06-08

2026-06-08 final pass: rechecked the temporally unstable 2025/2026 entries against official arXiv, OpenReview, Nature, ICLR, and PMLR pages. No BibTeX changes were required in this pass. The main live checks covered `maes2026lewm`, `maes2026stableworldmodel`, `huang2026vjepa`, `klindt2026lejepaworldmodel`, `usjepa2025`, `njepa2025`, `toso2026bisimjepa`, `assran2025vjepa2`, `vigmo`, `ghaemi2025seqjepa`, `voelcker2025calibratedvalueaware`, `hafner2025dreamerv3`, `dupuis2023vibr`, `gelada2019deepmdp`, `hansen2024tdmpc2`, and `bsmpc`.

Scope: all 41 citation keys used in `paper1/main.tex`. Unused BibTeX entries were removed from `paper1/references.bib`, so every remaining entry is cited.

| Key | Official source checked | Metadata conclusion | Text-use conclusion |
|---|---|---|---|
| `lecun2022path` | https://openreview.net/forum?id=BZ5a1r-kVsf | OK: Yann LeCun, 2022 OpenReview preprint. | OK: cited only for the JEPA latent-prediction framing. |
| `assran2023ijepa` | https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html | Fixed: added CVPR pages 15619--15629 and official CVF URL. | OK: I-JEPA image-domain statement matches the paper. |
| `bardes2024vjepa` | https://openreview.net/forum?id=QaCCuDfBk2 | Fixed: latest accepted TMLR version uses title without parenthetical and author `Mido Assran`; URL added. | OK: video-feature-prediction and V-JEPA statements match TMLR abstract. |
| `assran2025vjepa2` | https://arxiv.org/abs/2506.09985 | Fixed: added arXiv eprint, DOI, class, and URL. | OK: cited as V-JEPA 2 extending V-JEPA to understanding, prediction, and planning. |
| `maes2026lewm` | https://arxiv.org/abs/2603.19312 | Fixed: added arXiv eprint, DOI, class, and URL. | OK: LeWM two-loss end-to-end JEPA-from-pixels description matches the arXiv abstract. |
| `maes2026stableworldmodel` | https://openreview.net/forum?id=wjMSWSPNco | OK: ICLR 2026 Workshop on World Models Tiny Paper / arXiv metadata is current. | OK: cited for the stable-worldmodel baseline suite and benchmark ecosystem. |
| `sobal2025stresstesting` | https://openreview.net/forum?id=jON7H6A9UU | OK: WRL@ICLR 2025 poster metadata and URL match OpenReview. | OK: cited for the latent-dynamics planning baseline family. |
| `sobal2022jointembeddingpredictivearchitectures` | https://arxiv.org/abs/2211.10831 | OK: arXiv metadata, DOI, and URL match. | OK: cited for PLDM/JEPA slow-feature context. |
| `njepa2025` | https://arxiv.org/abs/2507.15216 | Fixed: removed non-official title parenthetical; added arXiv eprint, DOI, class, and URL. | OK: cited as JEPA robustness/noise-related work, consistent with diffusion-noise schedule claims. |
| `huang2026vjepa` | https://arxiv.org/abs/2601.14354 | Fixed: added arXiv eprint, DOI, class, and URL. | OK: Noisy-TV signal-recovery statement is aligned with the paper's reported JEPA-family `R^2 > 0.84` at high noise. |
| `usjepa2025` | https://arxiv.org/abs/2602.19322 | Fixed: added arXiv eprint, DOI, class, and URL. | OK: cited as JEPA robustness/domain-noise work for ultrasound representations. |
| `hafner2025dreamerv3` | https://www.nature.com/articles/s41586-025-08744-2 | Fixed: added Nature issue number and URL. | OK: cited for DreamerV3/world-model control. |
| `hansen2024tdmpc2` | https://openreview.net/forum?id=Oxh5CstDJU | Fixed: added official OpenReview URL. | OK: cited for scalable visual/continuous-control world models. |
| `vigmo` | https://openreview.net/forum?id=CoxruEzsd2 | OK: OpenReview ICLR 2026 submission metadata and URL match. | Fixed text: changed "sensor noise" to "unseen visual distractions" to match the official abstract. |
| `tamkin2023featuredropout` | https://proceedings.neurips.cc/paper_files/paper/2023/hash/c290d4373c495b2cad0625d6288260f0-Abstract-Conference.html | Fixed: expanded venue to NeurIPS, added volume 36 and official URL. | OK: cited for augmentation invariance/feature-destruction nuance. |
| `zhang2022rethinkaug` | https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Rethinking_the_Augmentation_Module_in_Contrastive_Learning_Learning_Hierarchical_Augmentation_CVPR_2022_paper.html | Fixed: added official CVF URL; pages already matched. | OK: cited for task-dependent augmentation invariances. |
| `roy2007effrank` | https://zenodo.org/records/40328 | Fixed: added 15th EUSIPCO wording, pages 606--610, DOI, and URL. | OK: cited for effective-rank diagnostic. |
| `jing2022dimcollapse` | https://openreview.net/forum?id=YevsQ05DEN7 | Fixed: added official OpenReview URL. | OK: cited for dimensional-collapse context. |
| `teoh2025nextlatent` | https://arxiv.org/abs/2511.05963 | Fixed: removed `Tim Pearce`, who is not in the official arXiv author list; added arXiv eprint, DOI, class, and URL. | OK: cited for next-latent compact world-model representation context. |
| `eppspulley1983` | https://academic.oup.com/biomet/article-pdf/70/3/723/687464/70-3-723.pdf | Fixed: added DOI and official Biometrika URL. | OK: cited for the empirical-characteristic-function normality test behind SIGReg. |
| `bardes2022vicreg` | https://openreview.net/forum?id=xm6YD62D1Ub | Fixed: added official OpenReview URL. | OK: cited for anti-collapse SSL regularization context. |
| `kornblith2019cka` | https://proceedings.mlr.press/v97/kornblith19a.html | Fixed: added ICML/PMLR volume, pages, publisher, and URL. | OK: cited for CKA representation similarity. |
| `alain2017linearprobes` | https://openreview.net/forum?id=HJ4-rAVtl | OK: ICLR 2017 workshop / arXiv metadata retained. | OK: cited for linear-probe diagnostics. |
| `sun2022knnood` | https://proceedings.mlr.press/v162/sun22d.html | Fixed: added ICML/PMLR volume, publisher, and URL; pages already matched. | OK: cited only as inspiration for nearest-neighbour latent scale. |
| `williams2007cem` | https://doi.org/10.1007/s11009-006-9753-0 | Fixed: added DOI URL. Visible metadata is OK: Kroese, Porotsky, Rubinstein, 2006. | OK: cited for CEM continuous optimization used in planning. |
| `garcia1989mpc` | https://doi.org/10.1016/0005-1098(89)90002-2 | Fixed: added DOI and DOI URL. | OK: cited for MPC background. |
| `wang2020alignuniform` | https://proceedings.mlr.press/v119/wang20k.html | Fixed: added ICML/PMLR volume, pages, publisher, and URL. | OK: cited for alignment/uniformity and augmentation-induced invariance. |
| `garrido2023rankme` | https://proceedings.mlr.press/v202/garrido23a.html | Fixed: added ICML/PMLR volume, pages, publisher, URL, and protected `RankMe` casing. | OK: cited for rank as label-free representation-quality diagnostic motivation. |
| `kostrikov2020drq` | https://openreview.net/forum?id=GY6-6sTvGaf | Fixed: added official OpenReview URL. | OK: cited for DrQ/data augmentation in pixel RL. |
| `yarats2022drqv2` | https://openreview.net/forum?id=_SJ-_yyes8 | Fixed: added official OpenReview URL. | OK: cited for DrQ-v2 visual continuous-control augmentation baseline. |
| `hansen2021soda` | https://doi.org/10.1109/ICRA48506.2021.9561103 | Fixed: added ICRA pages 13611--13617, DOI, and DOI URL. | OK: cited for SODA/DMC-GB visual robustness through soft data augmentation. |
| `ghaemi2025seqjepa` | https://openreview.net/forum?id=GKt3VRaCU1 | Fixed: retained NeurIPS 2025 OpenReview URL and avoided an unverified proceedings volume. | OK: cited for architectural handling of invariance/equivariance tension. |
| `toso2026bisimjepa` | https://arxiv.org/abs/2602.18639 | Fixed: added arXiv eprint, DOI, class, and URL. | OK: cited for Bisim-JEPA/control-relevant invariant visual representations for planning. |
| `vanassel2025jointembeddingreconstruction` | https://arxiv.org/abs/2505.12477 | OK: arXiv metadata, DOI, and URL match. | OK: cited for joint embedding reducing pressure to encode high-magnitude irrelevant features while still needing aligned augmentations/bias. |
| `klindt2026lejepaworldmodel` | https://arxiv.org/abs/2605.26379 | OK: arXiv metadata, DOI, and URL match. | OK: cited for LeJEPA latent-variable recovery/latent-planning theory. |
| `grimm2020valueequivalence` | https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html | OK: NeurIPS 2020 metadata and URL match. | OK: cited for value-equivalent models serving downstream planning/control. |
| `voelcker2025calibratedvalueaware` | https://proceedings.mlr.press/v267/voelcker25a.html | OK: ICML 2025 PMLR 267:61745--61768 metadata matches. | OK: cited for calibrated value-aware model learning. |
| `dupuis2023vibr` | https://proceedings.mlr.press/v232/dupuis23a.html | OK: CoLLAs 2023 PMLR 232:658--682 metadata matches. | OK: cited for view-invariant value functions/Bellman residuals rather than generic representation invariance. |
| `zhang2021dbc` | https://iclr.cc/virtual/2021/poster/2863 | OK: ICLR 2021 metadata and URL match. | OK: cited for bisimulation-based invariant RL representations without reconstruction. |
| `gelada2019deepmdp` | https://proceedings.mlr.press/v97/gelada19a.html | OK: ICML 2019 PMLR 97:2170--2179 metadata matches. | OK: cited for DeepMDP latent state/model representation learning. |
| `bsmpc` | https://openreview.net/forum?id=F07ic7huE3 | OK: ICLR 2025 OpenReview metadata and arXiv DOI match. | OK: cited for bisimulation-regularized MPC. |
