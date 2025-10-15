from pycocotools.coco import COCO
from openai import OpenAI
from tqdm import tqdm
import json
import re

API_KEY = ""
ANN_FILE = "/content/drive/MyDrive/GA-CLIP/data/coco/captions_train2014.json"
OUT_FILE = "/content/drive/MyDrive/GA-CLIP/data/llm_data_train.jsonl"
CLIENT = OpenAI(api_key=API_KEY, base_url="https://api.novita.ai/openai")
PROMPT = "### Task Description\nYou are given an image caption. Perform the following steps and follow the rules carefully:\n1. Identify key objects: List only the core word of each key object.\n2. Describe properties: For each key object, write its properties in a single sentence.\n3. Describe relations: For each pair of relevant key objects, write a sentence describing their relation.\n\n### Rules\n* You should extract only a few objects that play a key role in the caption, ignoring those of minor importance.\n* If there are several identical objects and each is described separately in the caption, consider each one individually when identifying key objects by adding a representative adjective, but do not add an adjective if there is only one instance of the core object.\n* For each property, mention only its corresponding key object and avoid referring to any other identified key objects, even if this leaves the sentence incomplete.\n* Do not mention any other key objects while describing the relation of a pair of relevant key objects.\n* Write the items of properties and relations in a key: value format, where the key is a key object for a property or two related key objects separated by an ampersand for a relation, and the value is a brief sentence.\n* All of your descriptions must be explicitly stated in the caption. \n* Write your answer in the same format as the example outputs and do not generate even a single extra word of explanation.\n\n### Example Input 1\nA horse is carrying a large load of hay, with two people sitting on it.\n\n### Example Output 1\nKey Objects:\n* horse\n* load\n* people\n\nProperties:\n* horse: The horse is carrying.\n* load: The load is large and consists of hay.\n* people: There are two people sitting.\n\nRelations:\n* horse & load: The horse is carrying the load.\n* horse & people: The people are sitting on the horse.\n\n### Example Input 2\nA yellow cat with wide-open eyes is on the left of a black dog with tilted head and curled ears, and a beautiful white cat with smile is on the right of the black dog.\n\n### Example Output 2\nKey Objects:\n* yellow cat\n* dog\n* white cat\n\nProperties:\n* yellow cat: The yellow cat has wide-open eyes.\n* dog: The dog is black with a tilted head and curled ears.\n* white cat: The white cat has a smile.\n\nRelations:\n* yellow cat & dog: The yellow cat is on the left of the dog.\n* dog & white cat: The white cat is on the right of the dog.\n\n### Example Input 3\na big red telephone booth that a man is standing in\n\n### Example Output 3\nKey Objects:\n* booth\n* man\n\nProperties:\n* booth: The booth is big, red, and contains a telephone.\n* man: The man is standing.\n\nRelations:\n* booth & man: The man is standing in the booth.\n\n### Example Input 4\nA woman preparing desserts covered in cream on top of the short kitchen table.\n\n### Example Output 4\nKey Objects:\n* woman\n* desserts\n* table\n\nProperties:\n* woman: The woman is preparing.\n* desserts: The desserts are covered in cream and are on top.\n* table: The table is short and is in the kitchen.\n\nRelations:\n* woman & desserts: The woman is preparing the desserts.\n* desserts & table: The desserts are on top of the table.\n\n### Caption\n[Caption]"


def generate(prompt):
    response = CLIENT.chat.completions.create(
        model="google/gemma-3-12b-it",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0,
    )
    return response.choices[0].message.content


def extract_data(img_id, caption, output):
    try:
        if (
            "Key Objects:" not in output
            or "Properties:" not in output
            or "Relations:" not in output
        ):
            return None
        key_objects_match = re.search(
            r"Key Objects:\n([\s\S]*?)\n\nProperties:", output
        )
        if not key_objects_match:
            return None
        key_objects = [
            line.strip("* ").strip()
            for line in key_objects_match.group(1).split("\n")
            if line.strip()
        ]
        properties_match = re.search(r"Properties:\n([\s\S]*?)\n\nRelations:", output)
        if not properties_match:
            return None
        properties_lines = [
            line.strip("* ").strip()
            for line in properties_match.group(1).split("\n")
            if line.strip()
        ]
        properties = {}
        for line in properties_lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                properties[key.strip()] = value.strip()
        relations_match = re.search(r"Relations:\n([\s\S]*)", output)
        if not relations_match:
            return None
        relations_lines = [
            line.strip("* ").strip()
            for line in relations_match.group(1).split("\n")
            if line.strip()
        ]
        relations = {}
        for line in relations_lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                relations[key.strip()] = value.strip()
        return {
            "image_id": img_id,
            "caption": caption,
            "key_objects": key_objects,
            "properties": properties,
            "relations": relations,
        }
    except Exception:
        return None


def main():
    coco = COCO(ANN_FILE)
    img_ids = coco.getImgIds()
    buffer = []
    counter = 0
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for img_id in tqdm(img_ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            captions = [a["caption"].strip() for a in anns if a.get("caption")]
            if not captions:
                continue
            caption = max(captions, key=len)
            prompt = PROMPT.replace("[Caption]", caption)
            output = extract_data(img_id, caption, generate(prompt))
            if not output:
                continue
            buffer.append(output)
            counter += 1
            if counter % 250 == 0:
                for item in buffer:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                buffer = []
        for item in buffer:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")