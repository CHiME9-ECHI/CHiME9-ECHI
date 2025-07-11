import torch
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from enhancement.registry import register_enhancement
from shared.core_utils import get_model
from shared.signal_utils import STFTWrapper, prep_audio


@register_enhancement("baseline")
class Baseline:
    def __init__(
        self,
        inference_dir: str,
        config_path: str,
        ckpt_path: str,
        audio_device: str,
        window_size: int,
        stride: int,
        torch_device: str,
    ):
        self.train_dir = Path(inference_dir).parent / f"train_{audio_device}"

        self.model_cfg = OmegaConf.load(config_path)

        self.stft = STFTWrapper(**self.model_cfg.input.stft, device=torch_device)
        self.stft = self.stft.to(torch_device)

        self.model = get_model(self.model_cfg, ckpt_path)
        self.model = self.model.to(torch_device)
        self.model.eval()

        self.window_samples = window_size * self.model_cfg.input.sample_rate
        rem = (self.window_samples - self.stft.n_fft) % self.stft.hop_length
        if rem > 0:
            self.window_samples += self.stft.hop_length - rem

        self.stride_samples = stride * self.model_cfg.input.sample_rate

        self.olap_samples = self.window_samples - self.stride_samples
        if self.olap_samples < 0:
            raise ValueError(
                f"Stride must be smaller than window size! Window: {window_size}, stride: {stride}"
            )
        elif self.olap_samples > 0:
            self.crossfade = torch.hann_window(
                self.olap_samples * 2, device=torch_device
            )

    def get_train_config(self):
        return OmegaConf.load(self.train_dir / "hydra/.hydra/config.yaml")

    def process_session(
        self,
        device_audio: torch.Tensor,
        device_fs: int,
        spkid_audio: torch.Tensor,
        spkid_fs: int,
        kwargs: Dict | None = None,
    ) -> torch.Tensor:

        device_audio = prep_audio(
            device_audio,
            device_fs,
            self.model_cfg.input.channels,
            self.model_cfg.input.sample_rate,
            self.model_cfg.input.rms,
            batched=False,
        )
        spkid_audio = prep_audio(
            spkid_audio.squeeze(0),
            spkid_fs,
            1,
            self.model_cfg.input.sample_rate,
            self.model_cfg.input.rms,
            batched=False,
        )

        spkid_input = self.stft(spkid_audio).unsqueeze(0)
        spkid_lens = torch.tensor([spkid_input.shape[2]])

        duration = device_audio.shape[-1]

        output = torch.zeros(duration, device=device_audio.device)

        with torch.no_grad():
            for start in tqdm(range(0, duration, self.stride_samples)):

                end = start + self.window_samples
                if end > duration:
                    end = duration

                window_size = end - start

                snippet = device_audio[..., start:end]
                snippet = prep_audio(
                    snippet,
                    self.model_cfg.input.sample_rate,
                    self.model_cfg.input.channels,
                    self.model_cfg.input.sample_rate,
                    self.model_cfg.input.rms,
                    batched=False,
                )

                rem = (window_size - self.stft.n_fft) % self.stft.hop_length
                if rem > 0:
                    # Pad signal to stop stft truncating it
                    pad_samples = self.stft.hop_length - rem
                    snippet = torch.nn.functional.pad(snippet, (0, pad_samples))

                snippet = self.stft(snippet).unsqueeze(0)

                den_snippet = self.model(snippet, spkid_input, spkid_lens).squeeze(0)
                den_snippet = self.stft.inverse(den_snippet).squeeze(0).squeeze(0)

                den_snippet = den_snippet[:window_size]

                if start > 0 and den_snippet.shape[-1] > self.olap_samples:
                    den_snippet[: self.olap_samples] *= self.crossfade[
                        : self.olap_samples
                    ]

                if end < duration and den_snippet.shape[-1] > self.olap_samples:
                    den_snippet[-self.olap_samples :] *= self.crossfade[
                        -self.olap_samples :
                    ]

                output[start:end] += den_snippet
        return output
