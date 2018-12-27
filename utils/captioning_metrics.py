import sys
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
LIBRARY_PATH = "./utils/coco-caption"
sys.path.append(LIBRARY_PATH)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import pickle as pkl
import re


# novel objects defined for MSCOCO
noc_objects = ['bus', 'bottle', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']


def score(gts, res, ids):
  origingts = gts
  originres = res
  tokenizer = PTBTokenizer()
  gts  = tokenizer.tokenize(gts)
  res = tokenizer.tokenize(res)
  """
  scorers = [
      (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
      (Meteor(),"METEOR"),
      (Rouge(), "ROUGE_L"),
      (Cider(), "CIDEr")]
  """
  scorers = [(Meteor(),"METEOR")]
  for scorer, method in scorers:
    score, scores = scorer.compute_score(gts, res)
    print("{:<14}:\t{:0.4f}".format(method, score))

  F1_score = F1(originres, origingts)
  avg = 0.0
  for noc_word in sorted(F1_score.keys()):
    print("{:<14}:\t{:0.4f}".format(noc_word, F1_score[noc_word]))
    avg +=  F1_score[noc_word]

  avg = avg / len(F1_score.keys())
  print("{:<14}:\t{:0.4f}".format("Average", avg))


def load_captions(generated_caption, meta):
  gt, res = {}, {}
  for v in meta:
    vname = v['vname']
    if vname not in gt.keys():
      gt[vname] = []
    gt[vname].append({'caption': v['desc']})

  for vname in generated_caption:
    res[vname] = [{"caption": " ".join(generated_caption[vname])}]
  return gt, res



def split_sent(sent):
  sent = sent.lower()
  sent = re.sub('[^A-Za-z0-9\s]+', '', sent)
  return sent.split()


def F1(generated_caption, gt):

  F1_score = {}

  for noc_word in noc_objects:
      novel_images = []
      nonNovel_images = []
      for vname in gt.keys():
        has_novel = False
        for caption in gt[vname]:
          if noc_word in split_sent(caption["caption"]):
            has_novel = True
            break

        if has_novel:
          novel_images.append(vname)
        else:
          nonNovel_images.append(vname)


      # true positive are sentences that contain match words and should
      tp = sum([1 for name in novel_images if noc_word in split_sent(generated_caption[name][0]["caption"])])
      # false positive are sentences that contain match words and should not
      fp = sum([1 for name in nonNovel_images if noc_word in split_sent(generated_caption[name][0]["caption"])])
      # false nagative are sentences that do not contain match words and should
      fn = sum([1 for name in novel_images if noc_word not in split_sent(generated_caption[name][0]["caption"])])

      # precision = tp/(tp+fp)
      if tp > 0:
        precision = float(tp)/(tp+fp)
        # recall = tp/(tp+fn)
        recall = float(tp)/(tp+fn)
        # f1 = 2* (precision*recall)/(precision+recall)
        F1_score[noc_word] = 2.0*(precision*recall)/(precision+recall)
      else:
        F1_score[noc_word] = 0.
  return F1_score
