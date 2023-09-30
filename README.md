# CosmoBot
Elbrus Bootcamp | Team project


## 🦸‍♂️Time
1. [Vika Ivanova](https://github.com/Vikaska031)
2. [Salman Chakaev](https://github.com/veidlink)


## 🎯 Task
Creating the CosmoBot Telegram bot, a personal assistant in dealing with skin imperfections and the fear of visiting a cosmetologist. Upload your photo, and the bot will detect imperfections and provide an approximate treatment plan for you. You can also view summarized reviews of cosmetic products.
## Tech stack 
**BeautifulSoup** for parsing cosmetic product data

**Pandas for data** storage and processing

**YOLO5** for imperfection detection

**Aiogram** for bot development

**YandexCloud** for deployment

**HuggingFace** for processing product reviews

## 📚 Libraries 

```typescript

import json
import re
import pandas as pd
import time
import logging                                        
import torch
import requests                                
from aiogram import Bot, Dispatcher, types
from bs4 import BeautifulSoup
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
```
	

## 🧠 Challenges

1. Data retrieval

2. Insufficient resources for large models

3. Aiogram library
 
 ## 📋Prospects and Plans
1. Product integration

2. Expansion of the database

3. Monetization of the product


