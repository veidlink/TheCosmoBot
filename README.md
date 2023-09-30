# CosmoBot
Elbrus Bootcamp | Team project


## 🦸‍♂️ Team
1. [Vika Ivanova](https://github.com/Vikaska031)
2. [Salman Chakaev](https://github.com/veidlink)


## 🎯 Task
Creating the CosmoBot Telegram bot, a personal assistant in dealing with skin imperfections and the fear of visiting a cosmetologist. Upload your photo, and the bot will detect imperfections and provide an approximate treatment plan for you. You can also view summarized reviews of cosmetic products.
## Tech stack 
- **BeautifulSoup** for parsing cosmetic product data
- **Pandas** for data storage and processing
- **YOLO5** for imperfection detection
- **HuggingFace** for processing product reviews
- **Aiogram** for bot development
- **YandexCloud** and **Docker** for deployment

## 📚 Libraries 

```typescript

import json
import re
import pandas as pd
import uuid
import time
import logging                                        
import torch
import requests
import asyncio                           
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
 
 ## 📋 Prospects and Plans
1. Product integration
2. Expansion of the database
3. Monetization of the product

## 📱 How to use the bot
1. Send /start command

<div align="center">
    <img src="https://github.com/veidlink/TheCosmoBot/assets/137414808/81a369d7-907f-4d56-b448-d18e0f2a5c62" width="300">
</div>

3. Press "Что это за приложение?🦄" if you want to get the bot's description, else press "Хочу скорее воспользоваться🤤" and send your photo. You'll get drugs we recommend for the detected problems.

<div align="center">
    <img src="https://github.com/veidlink/TheCosmoBot/assets/137414808/d00986c0-2523-45b2-81a3-60663a811513" width="300">
</div>

4. If you want to see summarized reviews on the provided list of medication, press "Получить отзывы на препарат". Wait a little and get the results.

<div align="center">
    <img src="https://github.com/veidlink/TheCosmoBot/assets/137414808/088a2c89-3578-4101-a6c3-5d5a61d7d0af" width="300">
</div>


