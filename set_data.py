from collections import defaultdict, Counter
import random
import json
import os

FILE_NAME = "data/llm_data_train.jsonl"
LM_OUT_FILE = "data/lm_train.jsonl"
CLIP_OUT_FILE = "data/clip_train.jsonl"


def filter_data(key_objects, properties, relations, caption):
    for obj in key_objects:
        if any(obj in other_obj for other_obj in key_objects if other_obj != obj):
            continue
        elif caption.count(obj) > 1:
            return True
    for prop_text in properties.values():
        key_objects_modified = [
            obj
            for obj in sorted(key_objects, key=len)
            if not any(other in obj for other in key_objects if other != obj)
        ]
        count = sum(1 for obj in key_objects_modified if obj in prop_text)
        if count >= 2:
            return True
    if set(properties.keys()) != set(key_objects):
        return True
    else:
        for prop_key, prop_value in properties.items():
            if ("no" in prop_value and "descri" in prop_value) or prop_value.count(
                prop_key
            ) != 1:
                return True
    if relations == {} or properties == {}:
        return True
    for pair, rel_text in relations.items():
        if " & " not in pair:
            return True
        a, b = pair.split(" & ")
        if (
            a == b
            or a not in key_objects
            or b not in key_objects
            or any(item in rel_text for item in key_objects if item != a and item != b)
            or a not in rel_text
            or b not in rel_text
        ):
            return True
    return False


def set_lm_train_data():
    with open(FILE_NAME, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    train_data = []
    c = 0
    for item in data:
        image_id = item["image_id"]
        caption = item["caption"].lower()
        key_objects = [obj.lower() for obj in item.get("key_objects")]
        properties = {k.lower(): v.lower() for k, v in item.get("properties").items()}
        relations = {k.lower(): v.lower() for k, v in item.get("relations").items()}
        skip_item = filter_data(key_objects, properties, relations, caption)
        if skip_item:
            continue
        c += 1
        train_data.append(
            {
                "image_id": image_id,
                "source": f"{caption} => key objects",
                "target": ", ".join(key_objects),
            }
        )
        train_data.append(
            {
                "image_id": image_id,
                "source": f"{caption} => relevant objects",
                "target": ", ".join(
                    [
                        f"({', '.join(sorted(pair.split(' & '), key=lambda x: caption.find(x)))})"
                        for pair in relations.keys()
                    ]
                ),
            }
        )
        for obj in key_objects:
            train_data.append(
                {
                    "image_id": image_id,
                    "source": f"{caption} => {obj}",
                    "target": properties[obj],
                }
            )
        for pair, rel_text in relations.items():
            objs = tuple(pair.split(" & "))
            train_data.append(
                {
                    "image_id": image_id,
                    "source": f"{caption} => ({objs[0]}, {objs[1]})",
                    "target": rel_text,
                }
            )
    with open(LM_OUT_FILE, "w", encoding="utf-8") as fout:
        for item in train_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(c)


def set_clip_train_data():
    with open(FILE_NAME, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    train_data = []
    _id = 0
    for item in data:
        image_id = item["image_id"]
        caption = item["caption"].lower()
        key_objects = [obj.lower() for obj in item.get("key_objects")]
        properties = {k.lower(): v.lower() for k, v in item.get("properties").items()}
        relations = {k.lower(): v.lower() for k, v in item.get("relations").items()}
        skip_item = filter_data(key_objects, properties, relations, caption)
        if skip_item:
            continue
        _id += 1
        train_data.append(
            {
                "id": _id,
                "image_id": image_id,
                "status": "Pos",
                "text": f"general caption: {caption}",
            }
        )
        for obj in key_objects:
            _id += 1
            key_object_id = _id
            train_data.append(
                {
                    "id": _id,
                    "image_id": image_id,
                    "status": "Pos",
                    "text": f"properties of {obj}: {properties[obj]}",
                }
            )
            _id += 1
            train_data.append(
                {
                    "id": _id,
                    "image_id": image_id,
                    "status": f"InterNeg-{key_object_id}",
                    "text": f"properties of {obj}: {caption}",
                }
            )
            for pair, rel_text in relations.items():
                if obj in pair:
                    _id += 1
                    train_data.append(
                        {
                            "id": _id,
                            "image_id": image_id,
                            "status": f"InterNeg-{key_object_id}",
                            "text": f"properties of {obj}: {rel_text}",
                        }
                    )
            for other_obj in key_objects:
                if other_obj != obj and (
                    (
                        len(key_objects) == 2
                        and (
                            len(list(relations.values())) == 0
                            or (
                                f"{obj} and {other_obj}"
                                not in list(relations.values())[0]
                                .replace("the ", "")
                                .replace("a ", "")
                                .replace("an ", "")
                                and f"{other_obj} and {obj}"
                                not in list(relations.values())[0]
                                .replace("the ", "")
                                .replace("a ", "")
                                .replace("an ", "")
                            )
                        )
                        and len(
                            list(
                                set(
                                    word
                                    for word in properties.get(obj).split()
                                    if len(word) >= 3 and word != "the"
                                )
                                & set(
                                    word
                                    for word in properties.get(other_obj).split()
                                    if len(word) >= 3 and word != "the"
                                )
                            )
                        )
                        < 2
                    )
                    or (
                        f"{obj} & {other_obj}" not in relations.keys()
                        and f"{other_obj} & {obj}" not in relations.keys()
                    )
                ):
                    _id += 1
                    train_data.append(
                        {
                            "id": _id,
                            "image_id": image_id,
                            "status": f"IntraNeg-{key_object_id}",
                            "text": f"properties of {obj}: {properties[other_obj].replace(other_obj, obj)}",
                        }
                    )
        for pair, rel_text in relations.items():
            _id += 1
            pair_id = _id
            objs = tuple(pair.split(" & "))
            objs = tuple(sorted(objs, key=lambda x: rel_text.find(x)))
            train_data.append(
                {
                    "id": _id,
                    "image_id": image_id,
                    "status": "Pos",
                    "text": f"relation between {objs[0]} and {objs[1]}: {rel_text}",
                }
            )
            _id += 1
            train_data.append(
                {
                    "id": _id,
                    "image_id": image_id,
                    "status": f"InterNeg-{pair_id}",
                    "text": f"relation between {objs[0]} and {objs[1]}: {caption}",
                }
            )
            for obj in key_objects:
                if obj in pair:
                    _id += 1
                    train_data.append(
                        {
                            "id": _id,
                            "image_id": image_id,
                            "status": f"InterNeg-{pair_id}",
                            "text": f"relation between {objs[0]} and {objs[1]}: {properties[obj]}",
                        }
                    )
            symmetric_words = [
                "near",
                "next",
                "close",
                "adjacent",
                "contact",
                "connect",
                "cluster",
                "touch",
                "facing",
                "opposite",
                "across",
                "adjoin",
                "side",
                "together",
                "with",
                "against",
            ]
            no_symmetric = all(word not in rel_text for word in symmetric_words)
            if no_symmetric and not (
                objs[0] in objs[1]
                or objs[1] in objs[0]
                or objs[0] not in rel_text
                or objs[1] not in rel_text
            ):
                _id += 1
                placeholder = "_TEMP_"
                swapped = rel_text.replace(objs[0], placeholder)
                swapped = swapped.replace(objs[1], objs[0])
                train_data.append(
                    {
                        "id": _id,
                        "image_id": image_id,
                        "status": f"IntraNeg-{pair_id}",
                        "text": f"relation between {objs[1]} and {objs[0]}: {swapped.replace(placeholder, objs[1])}",
                    }
                )
    grouped = defaultdict(list)
    for item in train_data:
        if item.get("status") == "Pos":
            grouped[int(item["image_id"])].append(item)
    to_remove_ids = {image_id for image_id, items in grouped.items() if len(items) > 12}
    train_data = [
        item for item in train_data if int(item["image_id"]) not in to_remove_ids
    ]
    with open(CLIP_OUT_FILE, "w", encoding="utf-8") as fout:
        for item in train_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
