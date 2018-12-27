import os, pdb, json
from copy import deepcopy
from pathlib import Path
import pickle as pkl


save_file_path = "mscoco/noc_coco_cap.json"


class CaptionData(object):
  def __init__(self, sentences, image):
    self.sentences = deepcopy(sentences)
    self.image = image['file_path']
    self.split = image['split']

def load_split_map(cur_dir):
    split_path = cur_dir / 'split_noc_coco.json'
    split_file = json.load(open(split_path, 'r'))
    split_map = {}
    split_map['train'] = {}
    split_map['val'] = {}
    split_map['test'] = {}

    for img in split_file['train']:
      split_map['train'][img] = 1
    for img in split_file['val']:
      split_map['val'][img] = 1
    for img in split_file['val_train']:
      split_map['val'][img] = 1      
    for img in split_file['test']:
      split_map['test'][img] = 1
    for img in split_file['test_train']:
      split_map['test'][img] = 1
    return split_map

def get_vocab(cap_vocab, cur_dir):
  print("Total vocab", len(cap_vocab))

  # coco detection vocab
  with open(cur_dir/"coco_detection_vocab.pkl", "rb") as f:
    det_vocab = pkl.load(f)

  det_vocabs = []

  # since the max label of coco detection is 90, however, it only has 80 classes.
  detection_class_to_idx = {} 
  for key in det_vocab:
    det_vocabs.append(det_vocab[key]["name"])
    detection_class_to_idx[int(key)] = len(det_vocabs) - 1

  # generate vocabs for lstm and detection part
  det_vocabs = det_vocabs
  lstm_vocabs = ['<pad>', '<bos>', '<eos>', '<unk>']
  for word in cap_vocab:
    if word not in det_vocab:
      lstm_vocabs.append(word)

  total_vocab = lstm_vocabs + det_vocabs
  ix_to_word, word_to_ix = {}, {}

  for idx, word in enumerate(total_vocab):
    ix_to_word[idx] = word
    word_to_ix[word] = idx

  # with open("coco_split.vocab", "wb") as f:
  #   pkl.dump({"lstm_vocab":lstm_vocab, "detcls_vocab":det_vocab, "ix_to_word":ix_to_word, "word_to_ix":word_to_ix}, f, protocol=2)

  return word_to_ix, ix_to_word, lstm_vocabs, det_vocabs, detection_class_to_idx


def main(cur_dir):
  cap_coco_path = cur_dir / 'cap_coco.json'
  with open(cap_coco_path, 'r') as cfile:
    cap_coco_data = json.load(cfile)
  length = len(cap_coco_data)
  all_words2idx = dict()
  for cap_data in cap_coco_data:
    for sentence in cap_data:
      for word in sentence:
        if word not in all_words2idx:
          all_words2idx[word] = len(all_words2idx)

  # load dic_coco
  dic_coco_path = cur_dir / 'dic_coco.json'
  with open(dic_coco_path, 'r') as cfile:
    dic_coco_data = json.load(cfile)
  images = dic_coco_data['images']
  assert len(images) == length, '{:} vs {:}'.format(len(images), length)

  split_map = load_split_map(cur_dir)
  word_to_ix, ix_to_word, lstm_vocabs, det_vocabs, detection_class_to_idx = get_vocab(all_words2idx, cur_dir)

  all_captions = {"train":[], "test":[], "word_to_ix":word_to_ix, "ix_to_word":ix_to_word, 
                  "lstm_vocabs":lstm_vocabs, "det_vocabs":det_vocabs, "detection_class_to_idx":detection_class_to_idx}

  for sentences, image in zip(cap_coco_data, images):
    img_id = str(image['id'])
    if img_id in split_map['val']: 
      continue
    phase = "train" if img_id in split_map['train'] else "test"

    if phase == "train":
      assert image["file_path"].find("train") != -1

    if phase == "test":
      assert image["file_path"].find("val") != -1 or image["file_path"].find("train") != -1, image["file_path"]

    for sentence in sentences:
      cap = {}
      cap["vname"] = Path(image["file_path"]).name
      cap["final_captions"] = sentence

      cap["caption_inputs"] = [word_to_ix[word] for word in sentence]
      cap["desc"] = " ".join(sentence)

      all_captions[phase].append(cap)


  print("Train sentences:", len(all_captions["train"]), "Test sentences", len(all_captions['test']))

  with open(save_file_path, "w") as f:
    json.dump(all_captions, f)
  print ('Output to', save_file_path)

if __name__ == '__main__':
  current_dir = Path(__file__).parent.resolve()
  main(current_dir)
