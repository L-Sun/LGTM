import asyncio
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
import os

import marshmallow_dataclass
import orjson
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from lgtm.utils.cwd import temporary_change_cwd

BASE_URL = "https://api.openai.com/v1"
API_KEY = "",


@dataclass
class PartMotion:
    action: str = field(default="idle")
    text: str = field(default="does nothing")


@dataclass
class BodyPartAnnotation:
    head: PartMotion = field(default_factory=PartMotion)
    torso: PartMotion = field(default_factory=PartMotion)
    left_arm: PartMotion = field(default_factory=PartMotion)
    right_arm: PartMotion = field(default_factory=PartMotion)
    left_leg: PartMotion = field(default_factory=PartMotion)
    right_leg: PartMotion = field(default_factory=PartMotion)


BodyPartAnnotation_Schema = marshmallow_dataclass.class_schema(BodyPartAnnotation)()

# class BodyPartAnnotation_JSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, BodyPartAnnotation):
#             return BodyPartAnnotation_Schema.dump(obj)

#         return super().default(obj)

# class BodyPartAnnotation_JSONDecoder(json.JSONDecoder):
#     def __init__(self):
#         super().__init__(object_hook=self.object_hook)

#     def object_hook(self, obj):
#         return BodyPartAnnotation_Schema.load(obj)


class BodyPartAnnotationTool:
    prompt = """
Prompt:
Decompose the given human motion description into JSON format, detailing the actions of specific body parts which group all joints. 
Some joint is important in the given motion description. If the action is about whole body, you need populate all body parts, such as dance.

Body Parts:
All body joints will be group to the following body part list
["head", "left_arm", "right_arm", "torso", "left_leg", "right_leg"]

Output Requirements:
- Format: JSON
- Exclude body parts without actions
- Only the key in Body Parts is allowed
- Do not include explanations or additional information
- Do not repeat keys

Terminology Adjustments:
- Replace "left_hand" with "left_arm"
- Replace "right_hand" with "right_arm"
- Replace "left_foot" with "left_leg"
- Replace "right_foot" with "right_leg"
- Replace "hip" with "torso"

Examples:

Input: A person is walking forward and waving their right hand twice.
Output:
{
    "right_arm": {
        "action": "wave",
        "text": "waves twice"
    },
    "left_leg": {
        "action": "walk",
        "text": "walks forward"
    },
    "right_leg": {
        "action": "walk",
        "text": "walks forward"
    }
}

Input: a person squats down and puts their hands above their head.
Output:
{
    "torso": {
        "action": "squat",
        "text": "squat down"
    },
    "left_arm": {
        "action": "raise",
        "text": "puts hands above head"
    },
    "right_arm": {
        "action": "raise",
        "text": "puts hands above head"
    },
    "left_leg": {
        "action": "squat",
        "text": "squat down"
    },
    "right_leg": {
        "action": "squat",
        "text": "squat down"
    }
}

Input: a person is balancing on something.
Output:
{
    "left_arm": {
        "action": "balance",
        "text": "balances with arms"
    },
    "right_arm": {
        "action": "balance",
        "text": "balances with arms"
    }
}

Input: a man is dancing
Output:
{
    "left_arm": {
        "action": "dance",
        "text": "dances"
    },
    "right_arm": {
        "action": "dance",
        "text": "dances"
    },
    "torso": {
        "action": "dance",
        "text": "dances"
    },
    "left_leg": {
        "action": "dance",
        "text": "dances"
    },
    "right_leg": {
        "action": "dance",
        "text": "dances"
    }
}
"""

    def __init__(self, api_key: str, base_url: str, model: str = "gpt-3.5-turbo-1106"):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def augment(self, seg_id: str, motion_description: str) -> dict[str, BodyPartAnnotation]:
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt
                    },
                    {
                        "role": "user",
                        "content": f"Input: {motion_description}"
                    },
                ],
                model=self.model,
                temperature=0.3,
                timeout=30,
            )
        except Exception as e:
            print(f"seg_id: {seg_id}")
            raise e

        try:
            json_content = self._remove_response_prefix_suffix(response.choices[0].message.content)
            body_part_annotation: BodyPartAnnotation = BodyPartAnnotation_Schema.loads(json_content)
        except Exception as e:
            print(f"seg_id: {seg_id}")
            print(f"Error: {e}")
            print(f"Content: {json_content}")
            raise e

        return {seg_id: body_part_annotation}

    def load_result(self, path: Path) -> dict[str, BodyPartAnnotation]:
        if not path.exists():
            with open(path, 'wb') as f:
                f.write(orjson.dumps(dict()))

        with open(path, 'rb') as f:
            body_part_annotation_json = orjson.loads(f.read())

        for key in body_part_annotation_json:
            body_part_annotation_json[key] = BodyPartAnnotation_Schema.load(body_part_annotation_json[key])

        return body_part_annotation_json

    def save_result(self, path: Path, result: dict[str, BodyPartAnnotation]):
        with open(path, 'wb') as f:
            f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))

    def _remove_response_prefix_suffix(self, json_str: str):
        return json_str.replace("Output:", "").replace("```json\n", "").replace("```", "")


async def batch_augment(original_annotations_path: Path, output_path: Path):
    tool = BodyPartAnnotationTool(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    body_part_group_annotations_dir = Path("datasets/annotations/humanml3d/body_part_group_annotations")
    with open(original_annotations_path, "r") as f:
        original_annotation_json: dict = orjson.loads(f.read())

        original_annotations = []
        for file_item in tqdm(original_annotation_json.values()):
            for annotation in file_item["annotations"]:
                original_annotations.append({
                    "seg_id": annotation["seg_id"],
                    "text": annotation["text"],
                })

    if not body_part_group_annotations_dir.exists():
        body_part_group_annotations_dir.mkdir(parents=True)

    group_size = 500
    for index in tqdm(range(0, len(original_annotations), group_size), position=0, desc="Body Part Augmentation"):
        body_part_group_annotations_path = body_part_group_annotations_dir / f"{index}_{index+group_size}.json"

        group_result = tool.load_result(body_part_group_annotations_path)

        group_original_annotations = list(filter(lambda annotation: annotation["seg_id"] not in group_result, original_annotations[index:index + group_size]))

        group_tasks = [tool.augment(seg_id=annotation["seg_id"], motion_description=annotation["text"]) for annotation in group_original_annotations]

        for body_part_annotation in tqdm_asyncio.as_completed(group_tasks, position=1, leave=False, desc=f"augment {index}:{index+group_size}"):
            try:
                group_result |= await body_part_annotation
            except Exception as e:
                print(f"Error: {e}")

        tool.save_result(body_part_group_annotations_path, group_result)

    # merge all group annotations
    result = tool.load_result(output_path)
    for group_result_path in tqdm(list(body_part_group_annotations_dir.glob("*.json")), desc="Merge"):
        group_result = tool.load_result(group_result_path)
        result |= group_result

    tool.save_result(output_path, result)


def split_for_tmr(original_annotations_path: Path, body_part_annotations_path: Path):
    tool = BodyPartAnnotationTool(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    result = tool.load_result(body_part_annotations_path)

    with open(original_annotations_path, "r") as f:
        original_annotation_json: dict = orjson.loads(f.read())

    # # copy to part
    for part_name in ["head", "left_arm", "right_arm", "torso", "left_leg", "right_leg"]:
        save_path = Path(f"datasets/annotations/body_part/humanml3d/{part_name}/annotations.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        for key_id in tqdm(original_annotation_json):
            for annotation in original_annotation_json[key_id]["annotations"]:
                seg_id = annotation["seg_id"]
                if seg_id in result:
                    annotation["text"] = asdict(result[seg_id])[part_name]["text"] # replace with part-level text

        with open(save_path, 'wb') as f:
            f.write(orjson.dumps(original_annotation_json))

        os.symlink(Path("../../../humanml3d/splits"), Path(f"./datasets/annotations/body_part/humanml3d/{part_name}/splits"), target_is_directory=True)


async def main():
    with temporary_change_cwd("third_packages/TMR"):
        original_annotations_path = Path("datasets/annotations/humanml3d/annotations.json")
        body_part_annotations_path = Path("datasets/annotations/humanml3d/body_part_annotations.json")

        parser = ArgumentParser()
        parser.add_argument("--only_split", type=bool, default=False)

        args = parser.parse_args()
        if not args.only_split:
            await batch_augment(original_annotations_path, body_part_annotations_path)

        split_for_tmr(original_annotations_path, body_part_annotations_path)


if __name__ == '__main__':
    asyncio.run(main())
