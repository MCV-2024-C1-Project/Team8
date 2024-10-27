import pickle
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.descriptors import GaborDescriptor
from src.paths import WEEK_3_RESULTS_PATH
from src.similarities import CosineSimilarity
import shutil
import os

def split_image_into_blocks(image: Image, num_blocks: int) -> list[Image]:
    width, height = image.size
    block_width, block_height = width // num_blocks, height // num_blocks
    return [image.crop(
                (col * block_width, row * block_height,
                (col + 1) * block_width, (row + 1) * block_height)
            ) for row in range(num_blocks) for col in range(num_blocks)]


def process_image_partitions(output_path: Path, image_list: list[Image],
                             partition_levels: list[int], mode: str = 'auto',
                             ) -> dict[int, list[list[Image]]]:
    partitioned_images = {level: [] for level in partition_levels}
    for level in partition_levels:
        level_output_dir = output_path.with_name(f"{output_path.stem}_level_{level}{output_path.suffix}")
        if mode == 'compute_notsave' and os.path.isdir(level_output_dir):
            shutil.rmtree(level_output_dir)

        if mode != 'compute' and level_output_dir.exists():
            for idx in tqdm(range(len(image_list)), desc=f"Loading images at level {level}"):
                partitions = []
                block_idx = 0
                while True:
                    img_path = level_output_dir / f"img_{idx}_block_{block_idx}.jpg"
                    if not img_path.exists():
                        break
                    with Image.open(img_path) as img:
                        partitions.append(img.copy())
                    block_idx += 1
                partitioned_images[level].append(partitions)
            continue

        level_output_dir.mkdir(parents=True, exist_ok=True)
        if level == 1:
            partitioned_images[level] = [[img] for img in image_list]
        else:
            partitioned_images[level] = [
                split_image_into_blocks(img, level)
                for img in tqdm(image_list, desc=f"Partitioning at level {level}")
            ]
        for img_idx, partitions in enumerate(partitioned_images[level]):
            for block_idx, block in enumerate(partitions):
                block.save(level_output_dir / f"img_{img_idx}_block_{block_idx}.jpg")

    return partitioned_images


def compute_partition_histograms(descriptors: list[GaborDescriptor],
                                 partition_levels: list[int],
                                 partitioned_images: dict[int, list[list[Image]]]
                                 ) -> dict[str, dict[int, dict[np.ndarray]]]:
    histograms = {}
    for descriptor in descriptors:
        histograms[descriptor.name] = {}
        for level in partition_levels:
            histograms[descriptor.name][level] = []
            for partitions in tqdm(partitioned_images[level], desc=f"Processing level {level}"):
                histograms_img = [descriptor.compute(np.array(partition)) for partition in partitions]
                concatenated_histogram = np.concatenate(histograms_img, axis=0)
                histograms[descriptor.name][level].append(concatenated_histogram)
    return histograms


def save_or_load_histograms(path: Path, compute_func: Callable, *args, load: bool = True) -> dict:
    if path.exists() and load:
        return load_histograms(path)
    histograms = compute_func(*args)
    with open(path, 'wb') as f:
        pickle.dump(histograms, f)
    return histograms


def load_histograms(file_path: Path) -> dict:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def calculate_query_distances(similarities: list[CosineSimilarity], descriptors: list[GaborDescriptor],
                              levels: list[int], db_histograms: dict[str, dict[int, list[np.ndarray]]],
                              query_histograms: dict[str, dict[int, list[np.ndarray]]]
                              ) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    distances = {}
    for similarity in similarities:
        distances[similarity.__class__.__name__] = {}
        for descriptor in descriptors:
            distances[similarity.__class__.__name__][descriptor.name] = {}
            for level in levels:
                db_hist = np.array(db_histograms[descriptor.name][level])
                query_hist = np.array(query_histograms[descriptor.name][level])
                distances[similarity.__class__.__name__][descriptor.name][level] = similarity.compute(query_hist,
                                                                                                      db_hist)
    return distances


def retrieve_top_k_indices(distances: np.ndarray, k: int) -> (list[list[int]], list[list[float]]):
    top_k_indices = np.argsort(distances, axis=1)[:, :k]
    top_k_similarities = np.take_along_axis(distances, top_k_indices, axis=1)
    return top_k_indices.tolist(), top_k_similarities.tolist()


def get_top_k_entries(distances_dict: dict[str, dict[str, dict[int, np.ndarray]]], k: int, verbose: bool = False) -> dict:
    top_k_results = {}
    for sim_name, desc_dict in distances_dict.items():
        top_k_results[sim_name] = {}
        for desc_name, levels in desc_dict.items():
            top_k_results[sim_name][desc_name] = {}
            for level, distances in levels.items():
                indices, sims = retrieve_top_k_indices(distances, k)
                top_k_results[sim_name][desc_name][level] = {"indexes": indices, "similarities": sims}
                if verbose:
                    print(f"{sim_name} - {desc_name} | Level {level}:")
                    print(f"Top-{k} Indices: {indices}\n")
    return top_k_results


def find_top_k_similar_images(query_images: list[Image], db_images: list[Image],
                              k: int = 5, level: int = 5,
                              descriptor = GaborDescriptor(), similarity_metric: CosineSimilarity = CosineSimilarity()
                              ) -> list[list[int]]:

    if any(isinstance(sublist, list) for sublist in query_images):
        flattened_query_images = [img for sublist in query_images for img in sublist]
    else:
        flattened_query_images = query_images

    if any(isinstance(sublist, list) for sublist in query_images):
        positions = [i for i, sublist in enumerate(query_images) for _ in sublist]
    else:
        positions = list(range(len(query_images)))

    query_partitions = process_image_partitions(WEEK_3_RESULTS_PATH / f"query_partitions",
                                                flattened_query_images, [level], mode="compute_notsave")
    db_partitions = process_image_partitions(WEEK_3_RESULTS_PATH / "db_partitions", db_images, [level])

    query_histograms = save_or_load_histograms(WEEK_3_RESULTS_PATH / f"query_histograms.pkl",
                                               compute_partition_histograms, [descriptor], [level], query_partitions,
                                               load=False)
    db_histograms = save_or_load_histograms(WEEK_3_RESULTS_PATH / "db_histograms.pkl",
                                            compute_partition_histograms, [descriptor], [level], db_partitions)

    distances = calculate_query_distances([similarity_metric], [descriptor], [level], db_histograms, query_histograms)
    retrieved_results = get_top_k_entries(distances, k)

    list_all_paintings = retrieved_results[similarity_metric.__class__.__name__][descriptor.name][level]["indexes"]

    list_all_images = [[] for _ in range(len(query_images))]

    for i, l in enumerate(list_all_paintings):
        index = positions[i]
        list_all_images[index].append(l)

    return list_all_images
