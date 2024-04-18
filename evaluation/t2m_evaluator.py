from pathlib import Path
from typing import Literal
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from lgtm.dataset.HumanML3D import HumanML3D, POS_enumerator


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size: int, pos_size: int, hidden_size: int, output_size: int):
        super(TextEncoderBiGRUCo, self).__init__()

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embeddings: Tensor, pos_one_hot: Tensor, text_lengths: Tensor) -> Tensor:
        num_samples = word_embeddings.shape[0]

        pos_embeddings = self.pos_emb(pos_one_hot)
        inputs = word_embeddings + pos_embeddings
        input_embeddings = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        emb = pack_padded_sequence(input_embeddings, text_lengths, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # shrink 4 times it just equal to Humanml3D unit_length

        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, motions: Tensor) -> Tensor:
        motions = motions.permute(0, 2, 1)
        outputs = self.main(motions).permute(0, 2, 1)
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MotionEncoderBiGRUCo, self).__init__()

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, movement_input: Tensor, motion_lengths: Tensor) -> Tensor:
        num_samples = movement_input.shape[0]

        input_embeddings = self.input_emb(movement_input)
        hidden = self.hidden.repeat(1, num_samples, 1)

        emb = pack_padded_sequence(input_embeddings, motion_lengths, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class T2M_Evaluator(nn.Module):
    def __init__(self, check_dir: Path, dataset_name: Literal["humanml", "kit"]):
        super().__init__()

        self.dim_pose = 263 if dataset_name == "humanml" else 251
        self.unit_length = HumanML3D.unit_length

        self.movement_encoder = MovementConvEncoder(
            self.dim_pose - 4,
            512,
            512,
        )
        self.motion_encoder = MotionEncoderBiGRUCo(
            512,
            1024,
            512,
        )
        self.text_encoder = TextEncoderBiGRUCo(
            300,
            len(POS_enumerator),
            512,
            512,
        )

        checkpoint_name = "t2m" if dataset_name == "humanml" else "kit"

        self.eval_mean = torch.from_numpy(np.load(check_dir / checkpoint_name / "Comp_v6_KLD01" / "meta" / "mean.npy")).to(torch.float32)
        self.eval_std = torch.from_numpy(np.load(check_dir / checkpoint_name / "Comp_v6_KLD01" / "meta" / "std.npy")).to(torch.float32)

        state_dict = torch.load(check_dir / checkpoint_name / "text_mot_match" / "model" / "finest.tar")

        self.movement_encoder.load_state_dict(state_dict["movement_encoder"])
        self.motion_encoder.load_state_dict(state_dict["motion_encoder"])
        self.text_encoder.load_state_dict(state_dict["text_encoder"])

        self.movement_encoder = self.movement_encoder
        self.motion_encoder = self.motion_encoder
        self.text_encoder = self.text_encoder

    def normalize(self, motion: Tensor) -> Tensor:
        return (motion - self.eval_mean.to(motion.device)) / self.eval_std.to(motion.device)

    def encode_motion(self, motions: Tensor, lengths: Tensor) -> Tensor:
        # since the t2m use GRU which need the motion data input to be collated with following steps:
        # 1. sort batch motions
        # 2. use pad_sequence
        # 3. use pack_padded_sequence in motion encoder, this function require the padded input is sorted descending
        # However, in our project, we dose assume the batch data is sorted,
        # so we need sort the motion by length before using T2M motion encoder,
        # then recover the sort of motion data after using it

        align_indices = torch.argsort(lengths, descending=True)

        motions = motions[align_indices]
        lengths = lengths[align_indices]

        movements = self.movement_encoder(motions[..., :-4])
        lengths = lengths // self.unit_length
        motion_embeddings: Tensor = self.motion_encoder(movements, lengths)

        align_back_indices = torch.argsort(align_indices)
        motion_embeddings = motion_embeddings[align_back_indices]

        return motion_embeddings

    def encode_text(self, word_embeddings: Tensor, pos_one_hot: Tensor, text_lengths: Tensor) -> Tensor:
        # same as the encode_motion we need sort the input by length and recover it after using text_encoder
        align_indices = torch.argsort(text_lengths, descending=True)

        word_embeddings = word_embeddings[align_indices]
        pos_one_hot = pos_one_hot[align_indices]
        text_lengths = text_lengths[align_indices]

        text_embeddings: Tensor = self.text_encoder(word_embeddings, pos_one_hot, text_lengths)

        align_back_indices = torch.argsort(align_indices)
        text_embeddings = text_embeddings[align_back_indices]

        return text_embeddings


if __name__ == "__main__":
    dataset = HumanML3D(Path("data/HumanML3D"), Path("data/glove"), "test")
    evaluator = T2M_Evaluator(Path("data/t2m_evaluator"), "humanml")
    data = dataset[0]

    x = evaluator.encode_motion(
        torch.from_numpy(data.motion).unsqueeze(0),
        torch.tensor([data.motion.shape[0]]),
    )
    y = evaluator.encode_text(
        torch.from_numpy(data.t2m_evaluation_info.word_embeddings).unsqueeze(0),
        torch.from_numpy(data.t2m_evaluation_info.pos_one_hots).unsqueeze(0),
        torch.tensor([data.t2m_evaluation_info.text_lengths]),
    )

    print(x.shape)
    print(y.shape)

    print(torch.cdist(x, y))
