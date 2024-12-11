from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.components.resize import Resizer
from birdset.datamodule.components.augmentations import PowerToDB
from birdset.datamodule.components.transforms import PreprocessingConfig
import birdset.utils as birdutils
from datasets import Dataset, DatasetDict, load_from_disk
from omegaconf import OmegaConf
from transformers import ConvNextForImageClassification
from torchaudio.transforms import Spectrogram, MelScale
import hydra
import pyrootutils
import torch.nn as nn
import torch

from util.event_extraction import load_audio

log = birdutils.get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["src"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base":None,
    "config_path": str(root / "configs/dataset_curation/gen_convnext_embeddings/"),
    "config_name": "HSN_local.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def generate_convnext_embeddings(cfg):
    log.info("Using config: \n%s", OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device {device}")
    
    # load data
    dataset = load_from_disk(cfg.dataset.data_dir)
    log.info("loaded dataset \n%s", dataset)

    # load convnext
    log.info("loading ConvNeXT")
    model = ConvNextForImageClassification.from_pretrained(
                "DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
                num_labels=cfg.dataset.num_classes,
                ignore_mismatched_sizes=True,
            )
    
    log.info("Deactivating classifying head")
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, input):
            return input
    
    model.classifier = Identity()

    log.info(f"Moving model to {device}")
    model = model.to(device)

    # convert samples to spectrogramms
    log.info("Configuring transforms")
    transformer = BirdSetTransformsWrapper(
            model_type=cfg.transforms.model_type,
            sampling_rate=cfg.dataset.sampling_rate,
            max_length=cfg.transforms.max_length,
            preprocessing=PreprocessingConfig(
                spectrogram_conversion= Spectrogram(
                    n_fft=cfg.transforms.preprocessor.n_fft,
                    hop_length=cfg.transforms.preprocessor.hop_length,
                    power=cfg.transforms.preprocessor.power,
                ),
                resizer=Resizer(
                    db_scale=True,
                    target_height=None,
                    target_width=None,
                ),
                melscale_conversion=MelScale(n_mels=cfg.transforms.preprocessor.mel_scale.n_mels, sample_rate=cfg.dataset.sampling_rate, n_stft=cfg.transforms.preprocessor.mel_scale.n_stft),
                dbscale_conversion=PowerToDB(),
                normalize_spectrogram=True,
                normalize_waveform=None,
                mean=cfg.transforms.preprocessor.mean,
                std=cfg.transforms.preprocessor.std,
            )
        )

    # map embeddings
    def get_embeddings(sample):
        audio, _ = load_audio(sample, min_len=cfg.sample_length, max_len=cfg.sample_length, sampling_rate=cfg.dataset.sampling_rate, pad_to_min_length=cfg.sample_padding)
        audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        spectogram = transformer._preprocess(audio, None).to(device)
        output = model.forward(spectogram)
        return output.logits.squeeze().detach()

    log.info("Generating and mapping ConvNeXT embeddings")
    embeddings_set = dataset.map(
        lambda sample: {"audio": get_embeddings(sample)},
        remove_columns=["filepath"])

    # save embeddings
    log.info(f"Saving data to {cfg.embeddings_save_path}")
    if isinstance(embeddings_set, DatasetDict):
        embeddings_set.save_to_disk(dataset_dict_path=cfg.embeddings_save_path)
    elif isinstance(embeddings_set, Dataset):
        embeddings_set.save_to_disk(dataset_path=cfg.embeddings_save_path)
    else:
        log.error(f"Saving of {embeddings_set.__class__} is not supported") 


if __name__ == "__main__":
    generate_convnext_embeddings()