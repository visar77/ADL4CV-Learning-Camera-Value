"""Download source GLBs for selected objects in G-buffer Objaverse."""

import argparse
import json
import random
import re
import shutil
import sys
import time
import gzip
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

INDEX_URL = (
    "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/"
    "gobjaverse_280k.json"
)
MAPPING_URL = (
    "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/"
    "gobjaverse_280k_index_to_objaverse.json"
)
GLB_BASE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
METADATA_BASE_URL = (
    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/"
)


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "ObjaverseRenderer downloader"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def load_index(index_path):
    if str(index_path).startswith(("http://", "https://")):
        return fetch_json(str(index_path))
    with Path(index_path).open(encoding="utf-8") as file:
        return json.load(file)


def object_ids(index):
    if not isinstance(index, list) or not all(isinstance(item, str) for item in index):
        raise ValueError("The dataset index must be a JSON list of object IDs.")
    return list(dict.fromkeys(index))


def select_objects(index, count=None, object_name=None, seed=None):
    candidates = object_ids(index)
    if object_name is not None:
        matches = [
            item
            for item in candidates
            if item == object_name
            or item.rsplit("/", 1)[-1] == object_name
            or object_name.lower() in item.lower()
        ]
        if not matches:
            raise ValueError(f"No object matching {object_name!r} exists in the index.")
        if len(matches) > 1:
            raise ValueError(
                f"Object name {object_name!r} is ambiguous; matches: "
                + ", ".join(matches[:10])
            )
        return matches

    if count is None or count < 1:
        raise ValueError("count must be at least 1.")
    if count > len(candidates):
        raise ValueError(
            f"Requested {count} objects, but the index contains only {len(candidates)}."
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def safe_filename(object_id):
    return object_id.strip("/").replace("/", "_")


def load_metadata(mapped_path, metadata_cache):
    shard = Path(mapped_path).parts[0]
    if shard not in metadata_cache:
        url = urljoin(METADATA_BASE_URL, f"{quote(shard, safe='')}.json.gz")
        request = Request(url, headers={"User-Agent": "ObjaverseRenderer downloader"})
        with urlopen(request, timeout=120) as response:
            with gzip.GzipFile(fileobj=response) as compressed:
                metadata_cache[shard] = json.load(compressed)

    uid = Path(mapped_path).stem
    try:
        name = metadata_cache[shard][uid]["name"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"No object name found in metadata for {uid!r}.") from error
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Object {uid!r} has no usable name in metadata.")
    return name.strip()


def safe_object_name(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name or "object"


def download_file(url, destination, retries=3):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return False

    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = Request(
                url, headers={"User-Agent": "ObjaverseRenderer downloader"}
            )
            with urlopen(request, timeout=120) as response, partial.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            partial.replace(destination)
            return True
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {error}") from error
            time.sleep(attempt)
    return False


def download_object(object_id, mapping, output_dir, metadata_cache, filename):
    try:
        mapped_path = mapping[object_id]
    except KeyError as error:
        raise ValueError(
            f"No source GLB mapping exists for G-buffer object {object_id!r}."
        ) from error
    if not isinstance(mapped_path, str) or not mapped_path.lower().endswith(".glb"):
        raise ValueError(
            f"Invalid source GLB mapping for {object_id!r}: {mapped_path!r}"
        )

    url = urljoin(GLB_BASE_URL, quote(f"glbs/{mapped_path}", safe="/"))
    name = load_metadata(mapped_path, metadata_cache)
    destination = output_dir / f"{filename or safe_object_name(name)}.glb"
    return download_file(url, destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download selected objects from G-buffer Objaverse."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--count", type=int, help="Randomly select this many objects."
    )
    selection.add_argument(
        "--object-name", help="Download one object ID or unique substring."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "objects",
        help="Destination directory (default: ./objects).",
    )
    parser.add_argument(
        "--index",
        default=INDEX_URL,
        help="Local index JSON or URL for the object list.",
    )
    parser.add_argument(
        "--mapping",
        default=MAPPING_URL,
        help="Local mapping JSON or URL from G-buffer IDs to source GLBs.",
    )
    parser.add_argument("--seed", type=int, help="Seed used for count-based selection.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index
    if index_path == INDEX_URL:
        cached_index = args.output_dir / "gobjaverse_280k.json"
        if cached_index.is_file():
            index_path = str(cached_index)
        else:
            print(f"Downloading index: {INDEX_URL}")
            with cached_index.open("w", encoding="utf-8") as file:
                json.dump(fetch_json(INDEX_URL), file)

    mapping = load_index(args.mapping)
    if not isinstance(mapping, dict):
        raise ValueError("The source GLB mapping must be a JSON object.")
    legacy_metadata_dir = args.output_dir / ".metadata"
    if legacy_metadata_dir.is_dir():
        shutil.rmtree(legacy_metadata_dir)

    selected = select_objects(
        load_index(index_path),
        count=args.count,
        object_name=args.object_name,
        seed=args.seed,
    )
    print(f"Selected {len(selected)} object(s): {', '.join(selected)}")
    metadata_cache = {}
    names = {}
    for object_id in selected:
        mapped_path = mapping.get(object_id)
        if isinstance(mapped_path, str):
            names[object_id] = safe_object_name(
                load_metadata(mapped_path, metadata_cache)
            )
    name_counts = {name: list(names.values()).count(name) for name in names.values()}
    for object_id in selected:
        print(f"Downloading {object_id}...")
        filename = names[object_id]
        if name_counts[filename] > 1:
            filename = f"{filename}__{safe_filename(object_id)}"
        downloaded = download_object(
            object_id, mapping, args.output_dir, metadata_cache, filename
        )
        status = "downloaded" if downloaded else "already exists"
        print(f"Finished {object_id}; {status} {filename}.glb.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
