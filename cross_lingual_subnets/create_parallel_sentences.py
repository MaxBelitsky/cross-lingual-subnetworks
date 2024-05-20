from datasets import load_dataset
import json

en_ur_dataset = load_dataset("open_subtitles", lang1="en", lang2="ur", split="train")
en_hi_dataset = load_dataset("open_subtitles", lang1="en", lang2="hi", split="train[:30000]")
# en_ru_dataset = load_dataset("open_subtitles", lang1="en", lang2="zh", split="train[:30000]")
print(en_ur_dataset["translation"][:10])

sentence_dict = {}
for i in range(100):
    sentence = {
        "en": en_ur_dataset["translation"][i]["en"],
        "ur": en_ur_dataset["translation"][i]["ur"],
        "hi": en_hi_dataset["translation"][i]["hi"]
    }
    sentence_dict[i] = sentence

print(sentence)

with open("data/text/parallel-sentences.json", "w") as f:
    json.dump(sentence_dict, f)

# print(en_ru_dataset["meta"][:10])
# TODO:
# download ewt.json
# put them through a model
# save encodings