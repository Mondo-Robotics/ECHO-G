# Licensing

ECHO-G uses separate licenses for code, data contributions and model weights.
The dataset and model/FGD weights are publicly available in separate Hugging Face repositories
at revision `v0.2.0`. See the [dataset download guide](DATASET.md) and
[weight download commands](../README.md#download-model-weights).

## Scope

| Material | Applicable license |
|---|---|
| Project code | [PolyForm Noncommercial 1.0.0](../LICENSE), with [NOTICE](../NOTICE) |
| ECHO-G contributions to processed robot motion, frozen conditions, annotations, splits, statistics, metadata and dataset documentation | [CC BY-NC 4.0](../LICENSE-DATA-WEIGHTS), to the extent the project holds the relevant rights |
| ECHO-G model checkpoints and project-trained FGD encoder weights designated for this release | [CC BY-NC 4.0](../LICENSE-DATA-WEIGHTS), covering the project's rights in the weights |
| BEAT2 source content, including clip-aligned audio, transcripts and source word times | Applicable upstream Apache-2.0 terms, attribution and modification notices retained in the dataset |
| G1 MuJoCo XML and meshes | Original BSD-3-Clause license included in the robot-asset archive |
| Other third-party source code and assets | Their respective licenses and notices |

CC BY-NC 4.0 permits sharing and adaptation for noncommercial purposes with attribution,
a license reference and an indication of changes. The [full license](../LICENSE-DATA-WEIGHTS)
controls. Noncommercial use is determined by the purpose of the use, not simply whether the
user is a company or a university. The project does not impose additional terms on material
already licensed by third parties.

SMPL-X model files and frozen encoder weights are not distributed in the dataset. Obtaining
or using those dependencies requires following their own terms. The dataset's
`THIRD_PARTY_NOTICES.md`, `MODIFICATIONS.md` and `licenses/` contain the included source
notices and license records. This project's license does not grant third-party rights beyond
those terms or establish a separate license for every generated output.

## Attribution and licensing contacts

Copyright (c) 2026 SOAR-LAB,
School of Intelligence Science and Technology, Nanjing University.

Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
Mondo Robotics (妙动科技).

Retain the supplied attribution, license and applicable notices when sharing these materials,
and identify your changes as required by the applicable license. Citation metadata is in
[CITATION.cff](../CITATION.cff).

For inquiries about commercial licensing of rights held by the project, contact both:

- Dr. Hao Xu: <xuhao3e8@gmail.com>
- Dr. Shuo Yang: <shuo.yang@mondorobotics.com>

These contacts cannot grant permissions on behalf of upstream rights holders. Any separate
commercial arrangement must cover the specific rights and dependencies involved.
