import pickle
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Literal, cast

import marshmallow_dataclass
import numpy as np
import orjson
import torch
from loguru import logger
from scipy.spatial.transform import Rotation
from torch import Tensor
from tqdm import tqdm

from lgtm.utils.typing import SequenceDataset


@dataclass
class HumanML3D_Annotation:
    id: str
    seg_id: str
    caption: str
    tokens: list[str]
    start: int
    end: int


HumanML3D_Annotation_Schema = marshmallow_dataclass.class_schema(HumanML3D_Annotation)()


@dataclass
class T2M_EvaluationInfo:
    pos_one_hots: np.ndarray
    word_embeddings: np.ndarray
    text_lengths: int


@dataclass
class HumanML3D_Data:
    motion: np.ndarray
    annotation: HumanML3D_Annotation
    t2m_evaluation_info: T2M_EvaluationInfo


class HumanML3D(SequenceDataset[HumanML3D_Data]):
    num_joints = 22 # number SMPL joint
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19], dtype=np.int32)

    # the data representation is following
    # (num_frames,                    1)  root rotation angular velocity at x-z plane
    # (num_frames,                    2)  root linear velocity at x-z plane
    # (num_frames,                    1)  root height
    # (num_frames, (num_joints - 1) * 3)  joints position in root space but added root height
    # (num_frames, (num_joints - 1) * 6)  joints rotation in parent space
    # (num_frames,  num_joints      * 3)  joint velocity in parent space
    # (num_frames,                    4)  foot contact
    feature_dim = 263

    unit_length = 4

    def __init__(self, data_dir: Path, glove_dir: Path, split: Literal["train", "test", "val", "all"]):
        """HumanML3D Dataset 
        
        Args:
            data_dir: the annotation dataset directory
            split: the split of dataset
            smplx_model_dir: the SMPL-X model directory
        """

        self.data_dir = data_dir / "HumanML3D"
        self.split = split

        self.mean: np.ndarray = np.load(self.data_dir / "Mean.npy", 'r')[...]
        self.std: np.ndarray = np.load(self.data_dir / "Std.npy", 'r')[...]

        self.annotations = self.load_annotations()
        self.seg_ids = list(self.annotations.keys())

        # T2M related
        self.word_vectorizer = WordVectorizer(glove_dir, "our_vab")
        self.max_text_length = 20

    def load_annotations(self) -> dict[str, HumanML3D_Annotation]:
        annotations_path = self.data_dir / f"{self.split}_annotations.json"
        annotations = dict[str, HumanML3D_Annotation]()

        if not annotations_path.exists():
            min_motion_len = 40

            def read_annotation(id: str):
                with open(self.data_dir / "texts" / f"{id}.txt") as f:
                    for seg_index, line in enumerate(f.readlines()):
                        data = line.strip().split("#")
                        caption = data[0]
                        tokens = data[1].split(" ")
                        start, end = float(data[2]), float(data[3])
                        start = 0.0 if np.isnan(start) else start
                        end = 0.0 if np.isnan(end) else end

                        start = int(20 * start)
                        end = int(20 * end)

                        yield HumanML3D_Annotation(id, f"{id}_{seg_index}", caption, tokens, start, end)

            with open(self.data_dir / f"{self.split}.txt", "r") as f:
                data_id = f.read().splitlines()

            for id in tqdm(data_id, "Read annotations"):
                if not (path := self.data_dir / "new_joint_vecs" / f"{id}.npy").exists():
                    continue

                for annotation in read_annotation(id):
                    if annotation.start == 0 and annotation.end == 0:
                        annotation.end = len(np.load(path, 'r'))

                    if min_motion_len <= (annotation.end - annotation.start) < 200:
                        annotations[annotation.seg_id] = annotation

            with open(annotations_path, 'wb') as f:
                f.write(orjson.dumps(annotations))

        with open(annotations_path, 'rb') as f:
            annotations_json = orjson.loads(f.read())
            for seg_id in annotations_json:
                annotations[seg_id] = cast(HumanML3D_Annotation, HumanML3D_Annotation_Schema.load(annotations_json[seg_id]))

        return annotations

    def de_normalize(self, normalized_motion: np.ndarray) -> np.ndarray:
        return normalized_motion * self.std + self.mean

    @staticmethod
    def recover(motion: np.ndarray) -> np.ndarray:
        def recover_root_position_rotation():
            root_angular_velocity = motion[:, 0]
            # the angle around y axis
            root_rotation_angle = np.zeros_like(root_angular_velocity)
            root_rotation_angle[1:] = root_angular_velocity[:-1]
            root_rotation_angle = np.cumsum(root_rotation_angle, axis=0)

            root_rotation_quat = Rotation.from_euler("y", root_rotation_angle)

            root_position = np.zeros((len(motion), 3))
            root_position[1:, [0, 2]] = motion[:-1, 1:3]
            root_position = root_rotation_quat.inv().apply(root_position)
            root_position = np.cumsum(root_position, axis=0)

            root_position[:, 1] = motion[:, 3]
            return root_position, root_rotation_quat

        positions = np.zeros((len(motion), HumanML3D.num_joints, 3), dtype=motion.dtype)
        positions[:, 1:] = motion[:, 4:4 + (HumanML3D.num_joints - 1) * 3].reshape(-1, HumanML3D.num_joints - 1, 3)

        root_position, root_rotation = recover_root_position_rotation()

        for joint in range(1, HumanML3D.num_joints):
            positions[:, joint] = root_rotation.inv().apply(positions[:, joint])

        positions[:, 0] = root_position                  # type:ignore
        positions[:, 1:, 0] += root_position[:, None, 0] # type:ignore
        positions[:, 1:, 2] += root_position[:, None, 2] # type:ignore

        return positions

    def get_by_seg_id(self, seg_id: str):
        annotation = self.annotations[seg_id]
        motion: np.ndarray = np.load(self.data_dir / "new_joint_vecs" / f"{annotation.id}.npy", 'r')[annotation.start:annotation.end]

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        new_length = len(motion)
        if coin2 == "double":
            new_length = (len(motion) // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            new_length = (len(motion) // self.unit_length) * self.unit_length

        offset = random.randint(0, len(motion) - new_length)
        motion = motion[offset:offset + new_length]

        motion = (motion - self.mean) / self.std

        return HumanML3D_Data(motion, annotation, self._generate_t2m_info(annotation))

    def __len__(self):
        return len(self.seg_ids)

    def __getitem__(self, index: int):
        return self.get_by_seg_id(self.seg_ids[index])

    def _generate_t2m_info(self, annotation: HumanML3D_Annotation):
        """Refer to Text2MotionDatasetV2 from other works like MDM"""

        tokens = annotation.tokens
        if len(tokens) < self.max_text_length:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            text_lengths = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_length + 2 - text_lengths)
        else:
            # crop
            tokens = tokens[:self.max_text_length]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            text_lengths = len(tokens)

        pos_one_hots = list[np.ndarray]()
        word_embeddings = list[np.ndarray]()
        for token in tokens:
            word_embedding, pos_one_hot = self.word_vectorizer[token]
            word_embeddings.append(word_embedding[None, :])
            pos_one_hots.append(pos_one_hot[None, :])

        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return T2M_EvaluationInfo(pos_one_hots, word_embeddings, text_lengths)


@dataclass
class BodyPart_HumanML3D_Data:
    motion: dict[str, np.ndarray]
    text: dict[str, str]

    original_data: HumanML3D_Data


class BodyPart_HumanML3D(SequenceDataset[BodyPart_HumanML3D_Data]):
    body_part_feature_dims = {
        "head": 24,
        "left_arm": 48,
        "right_arm": 48,
        "torso": 43,
        "left_leg": 50,
        "right_leg": 50,
    }

    def __init__(self, humanml3d: HumanML3D, body_part_annotations_path: Path):
        self.humanml3d = humanml3d

        with open(body_part_annotations_path, 'rb') as f:
            self.body_part_annotations = orjson.loads(f.read())

    def get_body_part_text(self, seg_id: str) -> dict[str, str]:
        if seg_id in self.body_part_annotations:
            return {part_name: self.body_part_annotations[seg_id][part_name]["text"] for part_name in self.body_part_feature_dims.keys()}
        else:
            return {part_name: self.humanml3d.annotations[seg_id].caption for part_name in self.body_part_feature_dims.keys()}

    def __len__(self):
        return len(self.humanml3d)

    def __getitem__(self, index: int) -> BodyPart_HumanML3D_Data:
        whole_data = self.humanml3d[index]

        return BodyPart_HumanML3D_Data(
            motion=rearrange_humanml3d_features_np(whole_data.motion),
            text=self.get_body_part_text(self.humanml3d.seg_ids[index]),
            original_data=whole_data,
        )


POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward', 'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn', 'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll', 'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily', 'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer:
    def __init__(self, meta_root: Path, prefix: str):
        """the data is from"""

        vectors: np.ndarray = np.load(meta_root / f'{prefix}_data.npy').astype(np.float32)
        words: list[str] = pickle.load(open(meta_root / f'{prefix}_words.pkl', 'rb'))
        word2idx: dict[str, int] = pickle.load(open(meta_root / f'{prefix}_idx.pkl', 'rb'))
        self.word2vec: dict[str, np.ndarray] = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_one_hot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator), dtype=np.float32)
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, token: str):
        word, pos = token.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_one_hot(vip_pos)
            else:
                pos_vec = self._get_pos_one_hot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_one_hot('OTHER')
        return word_vec, pos_vec


def rearrange_humanml3d_features_np(motion: np.ndarray) -> dict[str, np.ndarray]:
    num_joints = 22

    ric_data_offset = 4
    rot_data_offset = ric_data_offset + 3 * (num_joints - 1)
    local_vel_offset = rot_data_offset + 6 * (num_joints - 1)
    feet_contact_offset = local_vel_offset + 3 * num_joints

    assert feet_contact_offset + 4 == motion.shape[-1]

    def get_ric_rot_vel(part_index: list[int]):
        ric_data = np.concatenate([motion[..., ric_data_offset + 3 * (joint_index - 1):ric_data_offset + 3 * joint_index] for joint_index in part_index], axis=-1)
        rot_data = np.concatenate([motion[..., rot_data_offset + 6 * (joint_index - 1):rot_data_offset + 6 * joint_index] for joint_index in part_index], axis=-1)
        local_vel_data = np.concatenate([motion[..., local_vel_offset + 3 * joint_index:local_vel_offset + 3 * (joint_index + 1)] for joint_index in part_index], axis=-1)

        return np.concatenate([ric_data, rot_data, local_vel_data], axis=-1)

    head_part_index = [12, 15]
    left_arm_part_index = [13, 16, 18, 20]
    right_arm_part_index = [14, 17, 19, 21]
    torso_part_index = [0, 3, 6, 9]
    left_leg_part_index = [1, 4, 7, 10]
    right_leg_part_index = [2, 5, 8, 11]

    head_features = get_ric_rot_vel(head_part_index)
    left_arm_features = get_ric_rot_vel(left_arm_part_index)
    right_arm_features = get_ric_rot_vel(right_arm_part_index)
    torso_features = get_ric_rot_vel(torso_part_index[1:])
    left_leg_features = get_ric_rot_vel(left_leg_part_index)
    right_leg_features = get_ric_rot_vel(right_leg_part_index)

    torso_features = np.concatenate([torso_features, motion[..., 0:4], motion[..., local_vel_offset:local_vel_offset + 3]], axis=-1)
    left_leg_features = np.concatenate([left_leg_features, motion[..., feet_contact_offset:feet_contact_offset + 2]], axis=-1)
    right_leg_features = np.concatenate([right_leg_features, motion[..., feet_contact_offset + 2:feet_contact_offset + 4]], axis=-1)

    return {
        "head": head_features,
        "left_arm": left_arm_features,
        "right_arm": right_arm_features,
        "torso": torso_features,
        "left_leg": left_leg_features,
        "right_leg": right_leg_features,
    }


def rearrange_humanml3d_features(motion: Tensor) -> dict[str, Tensor]:
    num_joints = 22

    ric_data_offset = 4
    rot_data_offset = ric_data_offset + 3 * (num_joints - 1)
    local_vel_offset = rot_data_offset + 6 * (num_joints - 1)
    feet_contact_offset = local_vel_offset + 3 * num_joints

    assert feet_contact_offset + 4 == motion.shape[-1]

    def get_ric_rot_vel(part_index: list[int]):
        ric_data = torch.cat([motion[..., ric_data_offset + 3 * (joint_index - 1):ric_data_offset + 3 * joint_index] for joint_index in part_index], dim=-1)
        rot_data = torch.cat([motion[..., rot_data_offset + 6 * (joint_index - 1):rot_data_offset + 6 * joint_index] for joint_index in part_index], dim=-1)
        local_vel_data = torch.cat([motion[..., local_vel_offset + 3 * joint_index:local_vel_offset + 3 * (joint_index + 1)] for joint_index in part_index], dim=-1)

        return torch.cat([ric_data, rot_data, local_vel_data], dim=-1)

    head_part_index = [12, 15]
    left_arm_part_index = [13, 16, 18, 20]
    right_arm_part_index = [14, 17, 19, 21]
    torso_part_index = [0, 3, 6, 9]
    left_leg_part_index = [1, 4, 7, 10]
    right_leg_part_index = [2, 5, 8, 11]

    head_features = get_ric_rot_vel(head_part_index)
    left_arm_features = get_ric_rot_vel(left_arm_part_index)
    right_arm_features = get_ric_rot_vel(right_arm_part_index)
    torso_features = get_ric_rot_vel(torso_part_index[1:])
    left_leg_features = get_ric_rot_vel(left_leg_part_index)
    right_leg_features = get_ric_rot_vel(right_leg_part_index)

    torso_features = torch.cat([torso_features, motion[..., 0:4], motion[..., local_vel_offset:local_vel_offset + 3]], dim=-1)
    left_leg_features = torch.cat([left_leg_features, motion[..., feet_contact_offset:feet_contact_offset + 2]], dim=-1)
    right_leg_features = torch.cat([right_leg_features, motion[..., feet_contact_offset + 2:feet_contact_offset + 4]], dim=-1)

    return {
        "head": head_features,
        "left_arm": left_arm_features,
        "right_arm": right_arm_features,
        "torso": torso_features,
        "left_leg": left_leg_features,
        "right_leg": right_leg_features,
    }


def recover_rearranged_humanml3d_features_np(x: dict[str, np.ndarray]):
    num_joints = 22

    ric_data_offset = 4
    rot_data_offset = ric_data_offset + 3 * (num_joints - 1)
    local_vel_offset = rot_data_offset + 6 * (num_joints - 1)
    feet_contact_offset = local_vel_offset + 3 * num_joints

    assert feet_contact_offset + 4 == 263

    head_part_index = [12, 15]
    left_arm_part_index = [13, 16, 18, 20]
    right_arm_part_index = [14, 17, 19, 21]
    torso_part_index = [0, 3, 6, 9]
    left_leg_part_index = [1, 4, 7, 10]
    right_leg_part_index = [2, 5, 8, 11]

    result = np.empty((*x["head"].shape[:-1], 263))
    result[..., 0:4] = x["torso"][..., -7:-3]
    result[..., local_vel_offset:local_vel_offset + 3] = x["torso"][..., -3:]
    result[..., -4:-2] = x["left_leg"][..., -2:]
    result[..., -2:] = x["right_leg"][..., -2:]

    def set_ric_rot_vel(part_index: list[int], features: np.ndarray):
        _ric_data_offset = 0
        _rot_data_offset = _ric_data_offset + 3 * len(part_index)
        _local_vel_offset = _rot_data_offset + 6 * len(part_index)
        for index, joint_index in enumerate(part_index):
            result[..., ric_data_offset + 3 * (joint_index - 1):ric_data_offset + 3 * joint_index] = features[..., _ric_data_offset + 3 * index:_ric_data_offset + 3 * (index + 1)]
            result[..., rot_data_offset + 6 * (joint_index - 1):rot_data_offset + 6 * joint_index] = features[..., _rot_data_offset + 6 * index:_rot_data_offset + 6 * (index + 1)]
            result[..., local_vel_offset + 3 * joint_index:local_vel_offset + 3 * (joint_index + 1)] = features[..., _local_vel_offset + 3 * index:_local_vel_offset + 3 * (index + 1)]

    set_ric_rot_vel(head_part_index, x["head"])
    set_ric_rot_vel(left_arm_part_index, x["left_arm"])
    set_ric_rot_vel(right_arm_part_index, x["right_arm"])
    set_ric_rot_vel(torso_part_index[1:], x["torso"])
    set_ric_rot_vel(left_leg_part_index, x["left_leg"])
    set_ric_rot_vel(right_leg_part_index, x["right_leg"])

    return result


def recover_rearranged_humanml3d_features(x: dict[str, Tensor]):
    num_joints = 22

    ric_data_offset = 4
    rot_data_offset = ric_data_offset + 3 * (num_joints - 1)
    local_vel_offset = rot_data_offset + 6 * (num_joints - 1)
    feet_contact_offset = local_vel_offset + 3 * num_joints

    assert feet_contact_offset + 4 == 263

    head_part_index = [12, 15]
    left_arm_part_index = [13, 16, 18, 20]
    right_arm_part_index = [14, 17, 19, 21]
    torso_part_index = [0, 3, 6, 9]
    left_leg_part_index = [1, 4, 7, 10]
    right_leg_part_index = [2, 5, 8, 11]

    result = torch.empty(*x["head"].shape[:-1], 263, device=x["head"].device)
    result[..., 0:4] = x["torso"][..., -7:-3]
    result[..., local_vel_offset:local_vel_offset + 3] = x["torso"][..., -3:]
    result[..., -4:-2] = x["left_leg"][..., -2:]
    result[..., -2:] = x["right_leg"][..., -2:]

    def set_ric_rot_vel(part_index: list[int], features: Tensor):
        _ric_data_offset = 0
        _rot_data_offset = _ric_data_offset + 3 * len(part_index)
        _local_vel_offset = _rot_data_offset + 6 * len(part_index)
        for index, joint_index in enumerate(part_index):
            result[..., ric_data_offset + 3 * (joint_index - 1):ric_data_offset + 3 * joint_index] = features[..., _ric_data_offset + 3 * index:_ric_data_offset + 3 * (index + 1)]
            result[..., rot_data_offset + 6 * (joint_index - 1):rot_data_offset + 6 * joint_index] = features[..., _rot_data_offset + 6 * index:_rot_data_offset + 6 * (index + 1)]
            result[..., local_vel_offset + 3 * joint_index:local_vel_offset + 3 * (joint_index + 1)] = features[..., _local_vel_offset + 3 * index:_local_vel_offset + 3 * (index + 1)]

    set_ric_rot_vel(head_part_index, x["head"])
    set_ric_rot_vel(left_arm_part_index, x["left_arm"])
    set_ric_rot_vel(right_arm_part_index, x["right_arm"])
    set_ric_rot_vel(torso_part_index[1:], x["torso"])
    set_ric_rot_vel(left_leg_part_index, x["left_leg"])
    set_ric_rot_vel(right_leg_part_index, x["right_leg"])

    return result
