from birdset.modules.models.perch import PerchModel
import birdset.utils as birdutils
from datasets import Dataset, DatasetDict, load_from_disk
import hydra
from omegaconf import OmegaConf
import pyrootutils

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
    "config_path": str(root / "configs/dataset_curation/gen_perch_embeddings/"),
    "config_name": "HSN_local.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def generate_perch_embeddings(cfg):
    log.info("Using config: \n%s", OmegaConf.to_yaml(cfg))
    
    # load data
    dataset = load_from_disk(cfg.dataset.data_dir)
    log.info("loaded dataset \n%s", dataset)

    # load perch
    log.info("Instantiate Perch")
    perch = PerchModel(
        num_classes=cfg.dataset.num_classes,
        tfhub_version=""
    )

    # map embeddings
    def get_embeddings(sample):
        audio, _ = load_audio(sample, min_len=cfg.sample_length, max_len=cfg.sample_length, sampling_rate=cfg.dataset.sampling_rate, pad_to_min_length=cfg.sample_padding)
        return perch.get_embeddings(audio)[0]

    log.info("Generating and mapping Perch embeddings")
    embeddings_set = dataset.map(lambda sample: {"audio": get_embeddings(sample)}, remove_columns=["filepath"], keep_in_memory=True)

    # save embeddings
    embeddings_set.save_to_disk(dataset_dict_path=cfg.embeddings_save_path)

    # save embeddings
    log.info(f"Saving data to {cfg.embeddings_save_path}")
    if isinstance(embeddings_set, DatasetDict):
        embeddings_set.save_to_disk(dataset_dict_path=cfg.embeddings_save_path)
    elif isinstance(embeddings_set, Dataset):
        embeddings_set.save_to_disk(dataset_path=cfg.embeddings_save_path)
    else:
        log.error(f"Saving of {embeddings_set.__class__} is not supported") 


if __name__ == "__main__":
    generate_perch_embeddings()