## Environment Prepare

### Dependency
Create a virtual environment
```bash
python -m pip install --user virtualenv # install virtual environment manager
virtualenv --prompt lgtm .env           # create a virtual environment named lgtm
source .env/bin/activate                # activate it
```

Then install all dependency
```bash
pip install -r ./requirements.txt
```

### HumanML3D Data
```sh
cd third_package/HumanML3D
```
follow the `third_package/HumanML3D/README.md` to prepare data



### TMR prepare
The following steps is from `third_package/TMR/README.md` but add some steps for part-level TMR encoder
```bash
bash prepare_data_models.sh
```


#### (Optional) Generate part-level motion description from scratch
modify the config in `./third_packages/TMR/prepare/body_part_annotation_augmentation.py`
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


