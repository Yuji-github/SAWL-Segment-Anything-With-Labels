import json
import numpy as np
import os
from typing import List, Any
import py7zr


class BinaryMaskEncoder(json.JSONEncoder):
    """Special json encoder for numpy types
    This is for binary segmentation info
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_masks(selected_masks: List[dict[str, Any]], file_name: str) -> None:
    """Saving masks data as JSON in COCO format
    :param selected_masks:
    :param file_name:
    :return None:
    """
    if len(selected_masks) == 0:
        return

    if not os.path.exists("outputs_masks"):
        os.mkdir("outputs_masks")

    # file_name.split('.')[0]: removing extension format
    with open("outputs_masks/" + file_name.split(".")[0] + ".json", "w") as f:
        json.dump(selected_masks, f, cls=BinaryMaskEncoder)

    with py7zr.SevenZipFile("outputs_masks/" + file_name.split(".")[0] + ".7z", "w") as archive:
        archive.writeall(
            "outputs_masks/" + file_name.split(".")[0] + ".json", "outputs_masks/" + file_name.split(".")[0] + ".json"
        )

    os.remove("outputs_masks/" + file_name.split(".")[0] + ".json")
