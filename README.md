<!-- In this paper, we introduce LGTM, a novel Local-to-Global pipeline
for Text-to-Motion generation. LGTM utilizes a diffusion-based
architecture and aims to address the challenge of accurately translating textual descriptions into semantically coherent human motion in computer animation. Specifically, traditional methods often
struggle with semantic discrepancies, particularly in aligning specific motions to the correct body parts. To address this issue, we
propose a two-stage pipeline to overcome this challenge: it first
employs large language models (LLMs) to decompose global motion
descriptions into part-specific narratives, which are then processed
by independent body-part motion encoders to ensure precise local
semantic alignment. Finally, an attention-based full-body optimizer
refines the motion generation results and guarantees the overall -->

# LGTM: Local-to-Global Text-Driven Human Motion Diffusion Model
This repository contains the code and data for the paper "LGTM: Local-to-Global Text-Driven Human Motion Diffusion Model" (SIGGRAPH 2024).


## Setup and Data Preparation

### Virtual Environment
```bash
python -m pip install --user virtualenv # install virtual environment manager
virtualenv --prompt lgtm .env           # create a virtual environment named lgtm
source .env/bin/activate                # activate it
```

Then install all dependency
```bash
pip install -r ./requirements.txt
```

### Dependencies
```sh
cd third_package/HumanML3D
```
follow the `third_package/HumanML3D/README.md` to prepare original HumanML3D motion data.


Prepare TMR encoders and augmented part-level annotations.
```bash
bash prepare_data_models.sh
```


#### (Optional) Generate part-level motion description from scratch
You can generate part-level motion description by yourself. Firstly, you are required to provide OpenAI api key in `./third_packages/TMR/prepare/body_part_annotation_augmentation.py`
```python
BASE_URL = "https://api.openai.com/v1",
API_KEY = "",
```
Here, the `API_KEY` can get from your OpenAI account followed this [document](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
Then run the following commands:
```bash
cd third_packages/TMR
python prepare/body_part_annotation_augmentation.py
```
This script will use original full-body motion annotation for `third_packages/TMR/datasets/annotations/humanml3d/annotations.json`  as input, and generate part-level motion description for each body part. The output will be saved in `third_packages/TMR/datasets/annotations/humanml3d/body_part_annotations.json` and `third_packages/TMR/datasets/annotations/body_part`. The former output is group all part annotation in a map, the latter output is save each part annotation in a separate folder for train TMR encoders. You can use the `body_part_annotations.json` for what you want to do.


## Generation
The `playground.ipynb` notebook contains how to use LGTM to generate a motion from text. You can follow the steps in the notebook to generate a motion from text.


## Training
```bash
python -m lgtm.model.motion_diffusion fit --config configs/lgtm.yaml  --trainer.max_epoch=200
```