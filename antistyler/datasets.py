"""Convert INRIAPerson and PASCAL VOC2007 into the YOLO layout read by yolo_data.load_split.

Labels use the class indices of the COCO-pretrained Ultralytics detectors (Ultralytics' bundled
coco.yaml), so the default YOLO models can be attacked and evaluated on them directly. Images are
copied unchanged; boxes are converted from inclusive pixel indices to continuous coordinates.
"""

from collections import Counter
from pathlib import Path
import re
import shutil
from xml.etree import ElementTree

from .runtime import write_json
from .yolo_data import read_image

VOC_TO_COCO = {"aeroplane": "airplane", "diningtable": "dining table", "motorbike": "motorcycle",
               "pottedplant": "potted plant", "sofa": "couch", "tvmonitor": "tv"}
INRIA_TO_COCO = {"PASperson": "person"}
INRIA_SPLITS = {"train": "Train", "test": "Test"}
INRIA_FILENAME = re.compile(r'Image filename : "([^"]+)"')
INRIA_SIZE = re.compile(r"Image size \(X x Y x C\) : (\d+) x (\d+)")
INRIA_BOX = re.compile(r'Bounding box for object \d+ "([^"]+)" \(Xmin, Ymin\) - \(Xmax, Ymax\) : '
                       r"\((-?\d+), (-?\d+)\) - \((-?\d+), (-?\d+)\)")


def coco_names():
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    return {int(index): name for index, name in YAML.load(check_yaml("coco.yaml"))["names"].items()}


def inria_annotation(path):
    """(image path relative to the INRIA root, (width, height), [(class, x1, y1, x2, y2)])."""
    text = path.read_bytes().decode("latin-1")
    filename, size = INRIA_FILENAME.search(text), INRIA_SIZE.search(text)
    if filename is None or size is None:
        raise ValueError(f"{path}: missing image filename or size")
    # INRIA pixel indices are 0-based and inclusive.
    objects = [(INRIA_TO_COCO[name], int(x1), int(y1), int(x2) + 1, int(y2) + 1)
               for name, x1, y1, x2, y2 in INRIA_BOX.findall(text)]
    return filename.group(1), tuple(map(int, size.groups())), objects


def voc_annotation(path):
    """(image filename, (width, height), [(class, x1, y1, x2, y2)], difficult objects skipped)."""
    root = ElementTree.parse(path).getroot()
    size = root.find("size")
    objects, difficult = [], 0
    for item in root.iter("object"):
        if int(item.findtext("difficult", "0")):  # Ultralytics' VOC conversion also drops these
            difficult += 1
            continue
        box = item.find("bndbox")
        # VOC pixel indices are 1-based and inclusive.
        x1, y1, x2, y2 = (float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
        name = item.findtext("name").strip()
        objects.append((VOC_TO_COCO.get(name, name), x1 - 1, y1 - 1, x2, y2))
    return root.findtext("filename"), (int(size.findtext("width")), int(size.findtext("height"))), objects, difficult


def write_split(output, split, items, names):
    """items: (image path, (width, height), objects). Returns per-class object counts."""
    lookup = {name: index for index, name in names.items()}
    images, labels = output / split / "images", output / split / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    counts = Counter()
    for image, (width, height), objects in items:
        if tuple(read_image(image).shape[-2:]) != (height, width):
            raise ValueError(f"{image}: size differs from its annotation ({width} x {height})")
        lines = []
        for name, x1, y1, x2, y2 in objects:
            if name not in lookup:
                raise ValueError(f"{image}: class {name!r} is not a COCO class")
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{image}: empty box for {name!r}")
            lines.append(f"{lookup[name]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n")
            counts[name] += 1
        shutil.copyfile(image, images / image.name)
        (labels / f"{image.stem}.txt").write_text("".join(lines))
    return counts


def inria_splits(source):
    for split, directory in INRIA_SPLITS.items():
        items = []
        for annotation in sorted((source / directory / "annotations").glob("*.txt")):
            filename, size, objects = inria_annotation(annotation)
            items.append((source / filename, size, objects))
        yield split, items, 0


def voc_splits(source):
    lists = {"train": "train.txt", "val": "val.txt"}
    for split, listing in lists.items():
        ids = (source / "ImageSets/Main" / listing).read_text().split()
        yield split, *voc_items([(source / "Annotations" / f"{i}.xml", source / "JPEGImages") for i in ids])
    test = source / "Detection_splits/test_500"
    yield "test_500", *voc_items([(test / "annotations" / f"{image.stem}.xml", test / "images")
                                  for image in sorted((test / "images").glob("*.jpg"))])


def voc_items(entries):
    items, difficult = [], 0
    for annotation, images in entries:
        filename, size, objects, skipped = voc_annotation(annotation)
        items.append((images / filename, size, objects))
        difficult += skipped
    return items, difficult


def prepare_dataset(name, source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Directory is not empty: {output}. Choose a new directory.")
    splits = {"inria": inria_splits, "voc2007": voc_splits}[name](source)
    names = coco_names()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": name, "source": str(source), "class_names": "COCO (Ultralytics coco.yaml)",
               "splits": {}}
    for split, items, difficult in splits:
        counts = write_split(output, split, items, names)
        summary["splits"][split] = {"images": len(items), "objects": dict(sorted(counts.items())),
                                    "difficult_skipped": difficult}
        print(f"{split}: {len(items)} images, {sum(counts.values())} objects "
              f"({counts.get('person', 0)} person), {difficult} difficult skipped", flush=True)
    (output / "data.yaml").write_text(
        f"path: {output}\n" + "".join(f"{split}: {split}/images\n" for split in summary["splits"])
        + f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(names.items())))
    write_json(output / "conversion.json", summary)
    return summary
