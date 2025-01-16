from birdset.modules.models.perch import PerchModel
from birdset.datamodule.components.event_decoding import EventDecoding
from birdset.datamodule.components.feature_extraction import DefaultFeatureExtractor
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

    # configure decoding and padding
    log.info("Configure Decoder and Padding")
    decoder = EventDecoding(
    min_len=0,
    max_len=cfg.sample_length,
    sampling_rate=cfg.dataset.sampling_rate,
    extension_time=cfg.sample_length,
    extracted_interval=cfg.sample_length
    )
    
    extractor = DefaultFeatureExtractor(
        feature_size=1,
        sampling_rate=cfg.dataset.sampling_rate,
        padding_value=cfg.padding_value,
        return_attention_mask=False
    )


    # map embeddings
    def get_embeddings(batch):
        decoded_batch = decoder(batch)
        decoded_batch["audio"] = [audio_attribute["array"] for audio_attribute in decoded_batch["audio"]]
        samples = extractor(decoded_batch["audio"], padding="max_length", max_length=cfg.sample_length*cfg.dataset.sampling_rate, truncation=True, return_attention_mask=False)
        for b_idx in range(len(samples["input_values"])):
            decoded_batch["audio"][b_idx] = perch.get_embeddings(samples["input_values"][b_idx])[0]
        return decoded_batch

    log.info("Generating and mapping Perch embeddings")
    embeddings_set = dataset.map(get_embeddings, remove_columns=["filepath"], batched=True, batch_size=cfg.batch_size)

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