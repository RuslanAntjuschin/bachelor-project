from datasets import Dataset, DatasetDict, Audio
import birdset.utils as birdutils
import datasets
import hydra
from omegaconf import OmegaConf
import pyrootutils

log = birdutils.get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["src"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base":None,
    "config_path": str(root / "configs/dataset_curation/index_dataset/"),
    "config_name": "HSN_local.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def extract_and_index_dataset(cfg):
    log.info("Using config: \n%s", OmegaConf.to_yaml(cfg))
    
    # load data
    log.info(f"Loading data <{cfg.dataset.hf_path}/{cfg.dataset.hf_name}>")
    dataset = datasets.load_dataset(
        path=cfg.dataset.hf_path,
        name=cfg.dataset.hf_name,
        cache_dir=cfg.dataset.data_dir)
    log.info("loaded dataset \n%s", dataset)
    
    # event extraction
    if cfg.extract_events:
        log.info("Extracting Events")

        log.info(f"Instantiate event mapper <{cfg.mapper._target_}>")
        mapper = hydra.utils.instantiate(cfg.mapper)
        for split in cfg.extractable_splits:
            log.info(f"Extracting split \"{split}\"")

            dataset[split] = dataset[split].cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=32_000,
                mono=True,
                decode=False,
                ),
            )

            dataset[split] = dataset[split].map(
                mapper,
                batched=True,
                batch_size=300,
                num_proc=3,
                desc="event mapping",
                keep_in_memory=True,
            )
    else:
        log.info("Skipping Event Extraction")

    if cfg.removable_columns:
        log.info(f"Removing columns {cfg.removable_columns}")
        dataset = dataset.map(
            lambda sample: {},
            remove_columns=cfg.removable_columns,
            batched=True,
            batch_size=300,
            num_proc=3,
            desc="removing columns",
            keep_in_memory=True,
        )
    else:
        log.info("No columns to be removed found in config")

    # map indeces
    log.info("Indexing Data")
    indexed_dataset = dataset.map(lambda sample, idx: {"index": idx}, with_indices=True)

    # save embeddings
    log.info(f"Saving data to {cfg.indexed_save_path}")
    if isinstance(indexed_dataset, DatasetDict):
        indexed_dataset.save_to_disk(dataset_dict_path=cfg.indexed_save_path)
    elif isinstance(indexed_dataset, Dataset):
        indexed_dataset.save_to_disk(dataset_path=cfg.indexed_save_path)
    else:
        log.error(f"Saving of {indexed_dataset.__class__} is not supported") 

if __name__ == "__main__":
    extract_and_index_dataset()