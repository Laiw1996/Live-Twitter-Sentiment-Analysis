import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from emoji import UNICODE_EMOJI
import math
import emoji
import glob

def get_emoji():
	emoji_file = "./emoji&emoticons/emoji.txt"
	from pathlib import Path
	content = Path(emoji_file).read_text()

	emoji_pos = []
	emoji_neg = []

	for line in content.split("\n"):
	    if line == 'positive:':
	        positive = True
	    elif line == 'negative:':
	        positive = False;

	    if positive == True and line != 'positive:' and line != '':
	        emoji_pos.append(line)
	    elif positive == False and line != 'negative:' and line != '':
	        emoji_neg.append(line)
	return emoji_pos, emoji_neg

def is_emoji(twitter):
    count = 0
    for emoji in UNICODE_EMOJI:
        count += twitter.count(emoji)
    if count == 0:
        is_emoji=False
    else:
        is_emoji=True
    return is_emoji


def emoji_detect(twitter, emoji_pos, emoji_neg):
	positive = False
	negative = False
	for emo in emoji_pos:
	    if emo in twitter:
	        positive = True
	for emo in emoji_neg:
	    if emo in twitter:
	        negative = True

	if positive == True and negative == False:
	    result = 2
	elif positive == False and negative == True:
	    result = 0
	else:
		result = 100
	return result


def get_emoticon():
	emoticon_file="./emoji&emoticons/emoticons.txt"
	from pathlib import Path
	content = Path(emoticon_file).read_text()

	emoticons_pos = []
	emoticons_neg = []

	for line in content.split("\n"):
	    if line == 'positive:':
	        positive = True
	    elif line == 'negative:':
	        positive = False

	    if positive == True and line != 'positive:' and line != '':
	        emoticons_pos.append(line)
	    elif positive == False and line != 'negative:' and line != '':
	        emoticons_neg.append(line)
	return emoticons_pos, emoticons_neg


def emoticon_detect(twitter, emoticons_pos, emoticons_neg):
	positive = False
	negative = False
	for emo in emoticons_pos:
		if emo in twitter:
			positive = True
	for emo in emoticons_neg:
		if emo in twitter:
			negative = True

	if positive == True and negative == False:
	    result = 2
	elif positive == False and negative == True:
	    result = 0
	else:
		result = 100
	return result
