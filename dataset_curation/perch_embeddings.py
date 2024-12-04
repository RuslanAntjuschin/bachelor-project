from birdset.modules.models.perch import PerchModel
import birdset.utils as birdutils
import datasets
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
    "config_name": "local.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def generate_perch_embeddings(cfg):
    log.info("Using config: \n%s", OmegaConf.to_yaml(cfg))
    
    # instantiate datamodule
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    datamodule.prepare_data()
    dataset = datasets.load_from_disk(datamodule.disk_save_path)

    log.info("loaded dataset \n%s", dataset)

    # load perch
    log.info("Instantiate Perch")
    perch = PerchModel(
        num_classes=datamodule.num_classes,
        tfhub_version=""
    )

    # map embeddings
    def get_embeddings(sample):
        audio, _ = load_audio(sample, min_len=cfg.sample_length, max_len=cfg.sample_length, sampling_rate=cfg.datamodule.dataset.sampling_rate, pad_to_min_length=cfg.sample_padding)
        return perch.get_embeddings(audio)[0]

    log.info("Generating and mapping Perch embeddings")
    embeddings_set = dataset.map(lambda sample: {"audio": get_embeddings(sample)}, remove_columns=["filepath"])

    # save embeddings
    embeddings_set.save_to_disk(dataset_dict_path=cfg.embeddings_save_path)


if __name__ == "__main__":
    generate_perch_embeddings()