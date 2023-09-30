import json
import os
import uuid
import re
import pandas as pd
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

tokenizer = AutoTokenizer.from_pretrained("SiberiaSoft/SiberianFredT5-instructor")
summarizer = AutoModelForSeq2SeqLM.from_pretrained("SiberiaSoft/SiberianFredT5-instructor")

# Read the configuration file
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

# Access private info
TOKEN = config_data.get('TOKEN')
DEVICE = 'cpu' # CHANGE TO cpu IF NEEDED
summarizer.to(DEVICE)

def generate(prompt):
  data = tokenizer('<SC6>' + prompt + '\nОтвет: <extra_id_0>', return_tensors="pt")
  data = {k: v.to(DEVICE) for k, v in data.items()}
  output_ids = summarizer.generate(
      **data,  do_sample=True, temperature=0.2, max_new_tokens=512, top_p=0.95, top_k=5, repetition_penalty=1.03, no_repeat_ngram_size=2
  )[0]
  out = tokenizer.decode(output_ids.tolist())
  out = out.replace("<s>","").replace("</s>","")
  cleaned_text = re.sub(r'<[^>]+>', '', out)
  return cleaned_text

# Маппинг категорий на подходящие смайлики
category_smileys = {
    "концентрат": "😄",
    "крем или лосьон": "😊",
    "лосьон": "🌿",
    "скраб": "🌊",
    "гель": "😎",
    "маска": "😷",
    "сыворотка": "🧴",
    "тоник": "🌼",
    "кислота": "🍋",
    "пилинг": "🧼"
}

# Configure logging
logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)
             
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_first_name = message.from_user.first_name
    user_name = message.from_user.username
    logging.info(f'{user_first_name}, {user_name}, {time.asctime()}')

    # Создайте клавиатуру с двумя кнопками
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button1 = KeyboardButton("Что это за приложение?🦄")
    button2 = KeyboardButton("Хочу скорее воспользоваться🤤")

    markup.add(button1, button2)
    # Создайте меню с встроенными кнопками
    menu_markup = InlineKeyboardMarkup()
    menu_button1 = InlineKeyboardButton("Основное меню", callback_data="main_menu")
    menu_markup.add(menu_button1)

    # Отправьте приветственное сообщение с фото и клавиатурой с кнопками
    photo = open('botik.png', 'rb')  # Замените 'path_to_your_image.jpg' на путь к вашему изображению
    await bot.send_photo(
        chat_id=message.chat.id,
        photo=photo,
        caption='Привет❣️\n\nЯ твой косметический помощник. Чем могу помочь?',
        reply_markup=markup
    )
    photo.close()  # Закрываем файл с изображением

@dp.message_handler(lambda message: message.text == 'Что это за приложение?🦄')
async def app_info_handler(message: types.Message):
    # Создайте клавиатуру с кнопкой "Хочу скорее воспользоваться" после ответа
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button = KeyboardButton("Хочу скорее воспользоваться🤤")
    markup.add(button)
    # Создайте меню с встроенными кнопками
    menu_markup = InlineKeyboardMarkup()
    menu_button1 = InlineKeyboardButton("Основное меню", callback_data="main_menu")
    menu_markup.add(menu_button1)


    # Отправьте ответ после нажатия кнопки "Что это за приложение?" с клавиатурой с кнопкой
    await message.reply(
        "Я творение команды Пиксельные Титаны, создан, чтобы помогать людям определять проблемы на коже и примерную стоимость лечения. Очень важно, перед использованием рекомендаций обратиться к специалисту😷",
        reply_markup=markup
    )

@dp.message_handler(lambda message: message.text == 'Хочу скорее воспользоваться🤤')
async def start_over(message: types.Message):
    # Создайте клавиатуру с кнопкой "Что это за приложение" после ответа
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button = KeyboardButton("Что это за приложение?🦄")
    markup.add(button)
# Создайте меню с встроенными кнопками
    menu_markup = InlineKeyboardMarkup()
    menu_button1 = InlineKeyboardButton("Основное меню", callback_data="main_menu")
    menu_markup.add(menu_button1)

    # Отправьте ответ после нажатия кнопки "Хочу скорее воспользоваться тобой" с клавиатурой с кнопкой
    await message.reply(
        "Конечно, высылай фотографию! 😊",
        reply_markup=markup
    )

model = torch.hub.load(
'yolov5',  
'custom',  
path='datasets/weights/best_weights.pt', # путь к нашим весам
source='local',
force_reload=True 
);

model.to(DEVICE)
model.conf = 0.2 # устанавливаем порог
detected_labels = []

@dp.message_handler(content_types=['photo'])
async def send_drugs(message: types.Message):
    global detected_labels  # Reference the global variable
    user_id = message.from_user.id  
    user_first_name = message.from_user.first_name                              
    user_name = message.from_user.username
    
    # Get the file id from the message
    if message.content_type == 'photo':
        file_id = message.photo[-1].file_id
    elif message.content_type == 'document':
        file_id = message.document.file_id

    logging.info(f'{user_first_name}, {user_name} sent img at {time.asctime()}')

    # Get the File object from the bot
    try:
        file = await bot.get_file(file_id)
    except Exception as e:
        logging.error(f'Failed to get File object: {e}')
        return

    # Check if the File object is valid
    if not file:
        logging.error(f'File does not exist or is not accessible')
        return

    # Generate a unique filename for the downloaded file
    unique_filename = str(uuid.uuid4()) + '.jpg'

    # Download the file using the File object
    try:
        await bot.download_file(file.file_path, unique_filename)
    except Exception as e:
        logging.error(f'Failed to download file: {e}')
        return

    # Open the image using the same file name
    img = Image.open(unique_filename)
    # logging.info(f'Model got this: {type(img)}')
    results = model(img)
    detected_labels = [results.names[int(pred[-1])] for pred in results.pred[0]]
    detected_labels = list(set(detected_labels))
    logging.info(f'Были обнаружены: {detected_labels}')

    result = ''
    if len(detected_labels)==1:
        if detected_labels[0] != 'hyperceratos':
            if detected_labels[0] in ['papula', 'postacne', 'pustula']:
                result += '\n\nУ Вас папулы/постакне/пустулы ‼️'
                csv_file = "datasets/clean/acne_clean_with_categories.csv"
            elif detected_labels[0] == "cuperos":
                result += '\n\nУ Вас есть купероз ‼️'
                csv_file = "datasets/clean/cuperoz_clean_with_categories.csv" 
            elif detected_labels[0] == 'camedon':
                result += '\n\nУ Вас есть камедоны ‼️'
                csv_file = "datasets/clean/camedons_clean_with_categories.csv"

            try:
                data = pd.read_csv(csv_file)
                drug_names = {}
                drug_prices = {}
                drug_ratings = {}

                for x in data['category'].unique():
                    category_data = data[data['category'] == x]
                    top_3_names = category_data['drug_name'][:3].tolist()
                    top_3_prices = category_data['price'][:3].tolist()
                    top_3_ratings = category_data['rating'][:3].tolist()
                    
                    drug_names[x] = top_3_names
                    drug_prices[x] = top_3_prices
                    drug_ratings[x] = top_3_ratings

                result += "\n\nВам рекомендовано:"

                for category, names, prices, ratings in zip(drug_names.keys(), drug_names.values(), drug_prices.values(), drug_ratings.values()):
                    smiley = category_smileys.get(category, "😃")  # По умолчанию используется "😃"
                    result += f"\n\n{smiley} Категория: {category}"
                    for i, (name, price, rating) in enumerate(zip(names, prices, ratings), start=1):
                        result += f'\n{i}. {name} (рейтинг: {rating}, цена: {price}Р)'

                    
            except Exception as ex:
                result += f'\nОшибка: {ex}'

        else:
            result += '\n\nУ Вас есть гиперкератоз ‼️'
            csv_file = 'datasets/clean/kislots_and_pilings.csv'
            data = pd.read_csv(csv_file)
            kislots_names = data[data['category'] =='кислота']['drug_name'].tolist()
            pilings_names = data[data['category'] =='пилинг']['drug_name'].tolist()

            try: 
                top_3_kislots = kislots_names[:3]
                top_3_pilings = pilings_names[:3]

                top_3_kislots_r = data[data['category'] =='кислота']['rating'].tolist()[:3]
                top_3_kislots_p = data[data['category'] =='кислота']['price'].tolist()[:3]

                top_3_pilings_r = data[data['category'] =='пилинг']['rating'].tolist()[:3]
                top_3_pilings_p = data[data['category'] =='пилинг']['price'].tolist()[:3]

                smiley = category_smileys.get('кислота', "😃")
                result += f'\n\n{smiley} Советуем одну из кислот:'
                for i, (name, rating, price) in enumerate(zip(top_3_kislots, top_3_kislots_r, top_3_kislots_p), start=1):
                    result += f'\n{i}. {name} (рейтинг: {rating}, цена: {price}Р)'
                
                smiley = category_smileys.get('пилинг', "😃")
                result += f'\n\n{smiley} Из пилингов:'
                for i, (name, rating, price) in enumerate(zip(top_3_pilings, top_3_pilings_r, top_3_pilings_p), start=1):
                    result += f'\n{i}. {name} (рейтинг: {rating}, цена: {price}Р)'
                    
            except Exception as ex:
                result += f'\nОшибка: {ex}'


    elif len(detected_labels)>=2:
        detected_labels = ['postacne' if x in ['papula', 'pustula'] else x for x in detected_labels]
        detected_labels = list(set(detected_labels))
        for problem in detected_labels:
            if problem != 'hyperceratos':
                if problem in ['papula', 'postacne', 'pustula']:
                    result += '\n\nУ Вас папулы/постакне/пустулы ‼️'
                    csv_file = "datasets/clean/acne_clean_with_categories.csv"
                elif problem == "cuperos":
                    result += '\n\n У Вас есть купероз ‼️'
                    csv_file = "datasets/clean/cuperoz_clean_with_categories.csv" 
                elif problem == 'camedon':
                    result += '\n\nУ Вас есть камедоны ‼️'
                    csv_file = "datasets/clean/camedons_clean_with_categories.csv"

                try:
                    data = pd.read_csv(csv_file)
                    drug_names = {}
                    drug_prices = {}
                    drug_ratings = {}

                    for x in data['category'].unique():
                        category_data = data[data['category'] == x]
                        top_3_names = category_data['drug_name'][:3].tolist()
                        top_3_prices = category_data['price'][:3].tolist()
                        top_3_ratings = category_data['rating'][:3].tolist()
                        
                        drug_names[x] = top_3_names
                        drug_prices[x] = top_3_prices
                        drug_ratings[x] = top_3_ratings

                    result += "\n\nВам рекомендовано:"

                    for category, names, prices, ratings in zip(drug_names.keys(), drug_names.values(), drug_prices.values(), drug_ratings.values()):
                        smiley = category_smileys.get(category, "😃")  # По умолчанию используется "😃"
                        result += f"\n\n{smiley} Категория: {category}"
                        for i, (name, price, rating) in enumerate(zip(names, prices, ratings), start=1):
                            result += f'\n{i}. {name} (рейтинг: {rating}, цена: {price}Р)'
                    
                except Exception as ex:
                    result += f'\nОшибка: {ex}'
                    
            else:
                result += '\n\nУ Вас есть гиперкератоз ‼️'
                csv_file = 'datasets/clean/kislots_and_pilings.csv'
                data = pd.read_csv(csv_file)
                kislots_names = data[data['category'] =='кислота']['drug_name'].tolist()
                pilings_names = data[data['category'] =='пилинг']['drug_name'].tolist()

                try: 
                    top_3_kislots = kislots_names[:3]
                    top_3_pilings = pilings_names[:3]

                    top_3_kislots_r = data[data['category'] =='кислота']['rating'].tolist()[:3]
                    top_3_kislots_p = data[data['category'] =='кислота']['price'].tolist()[:3]

                    top_3_pilings_r = data[data['category'] =='пилинг']['rating'].tolist()[:3]
                    top_3_pilings_p = data[data['category'] =='пилинг']['price'].tolist()[:3]
                    
                    smiley = category_smileys.get('кислота', "😃")
                    result += f'\n\n{smiley} Советуем одну из кислот:'
                    for i, (name, rating, price) in enumerate(zip(top_3_kislots, top_3_kislots_r, top_3_kislots_p), start=1):
                        result += f'\n{i}. {name} (рейтинг: {rating}, цена: {price}Р)'
                    
                    smiley = category_smileys.get('пилинг', "😃")
                    result += f'\n\n{smiley} Из пилингов:'
                    for i, (name, rating, price) in enumerate(zip(top_3_pilings, top_3_pilings_r, top_3_pilings_p), start=1):
                        result += f'\n{i}. {name} (рейтинг: {rating}, цена: {price}Р)'
                        
                except Exception as ex:
                    result += f'\nОшибка: {ex}'
    else:
        result = 'У вас не обнаружено проблемок, все хорошо 🦄'
    await bot.send_message(user_id, result)

    try:
        os.remove(unique_filename)
        logging.info(f'File {unique_filename} has been deleted.')
    except Exception as e:
        logging.error(f'Failed to delete file: {e}')

    if result != 'У вас не обнаружено проблемок, все хорошо 🦄':
        # Создание inline-кнопок
        get_reviews_button = InlineKeyboardButton(text="Получить отзывы на препараты", callback_data="get_reviews")
        keyboard = InlineKeyboardMarkup().add(get_reviews_button)
        # Отправка сообщения с кнопками
        await message.reply("Выберите действие:", reply_markup=keyboard)
    else:
        pass
   
@dp.callback_query_handler(text="get_reviews")
async def get_reviews(callback_query: types.CallbackQuery):
    user_id = callback_query.from_user.id
    await bot.send_message(user_id, 'Немного терпения... ⏳')
    global detected_labels  # Reference the global variable
    headers = requests.utils.default_headers()
    
    if len(detected_labels)==1:
        detected_labels = detected_labels[0]
        # await message.reply(detected_labels)
        if detected_labels != 'hyperceratos':
            if detected_labels in ['papula', 'postacne', 'pustula']:
                data = pd.read_csv("datasets/clean/acne_clean_with_categories.csv")
            elif detected_labels == "cuperos":
                data = pd.read_csv("datasets/clean/cuperoz_clean_with_categories.csv")
            elif detected_labels == 'camedon':
                data = pd.read_csv("datasets/clean/camedons_clean_with_categories.csv")

            try:
                all_drug_responses = {}
                for x in data['category'].unique():
                    result = ''
                    category_data = data[data['category'] == x]
                    top_3_names = category_data['drug_name'][:3].tolist()            
                    urls = category_data['page_url'][:3]
                    drug_responses = {}
        
                    for name, url in zip(top_3_names, urls):
                        url = 'https://'+url
                    # Отправляем HTTP запрос и получаем ответ
                        await asyncio.sleep(1)
                        r = requests.get(url, headers=headers)
                        soup = BeautifulSoup(r.text, 'html.parser')
                        drug_descs = soup.find('div', 'kr_review_plain_text').text
                        prompt_review  = f'{drug_descs[:3000]} \n\nВыдели главную мысль из всех отзывов. Ответ начни со слов: "Этот препарат...'
                        response_text = generate(prompt_review)
                        drug_responses[name] = response_text
                        
                    all_drug_responses[x] = drug_responses                  

                for cat, drugnamesNreviews in all_drug_responses.items():
                    smiley = category_smileys.get(cat, "😃")  # По умолчанию используется "😃"
                    result = f"\n\n{smiley} Категория: {cat}"
                    for i, (name, drug_r3views) in enumerate(drugnamesNreviews.items(), start=1):
                        result += f'\n\n{i}. {name}. \n\n{drug_r3views}'
                    await bot.send_message(user_id, result)                    
                    
            except Exception as ex:
                result = f'\nОшибка при получении отзывов с сайта. Пожалуйста, попробуйте еще раз.'
                await bot.send_message(user_id, result)

        else:
            data = pd.read_csv('datasets/clean/kislots_and_pilings.csv')
            kislots_names = data[data['category'] =='кислота']['drug_name'].tolist()
            pilings_names = data[data['category'] =='пилинг']['drug_name'].tolist()

            try:
                result = ''
                pilings_reviews = {}
                kislots_reviews = {}
                top_3_kislots = kislots_names[:3]
                top_3_pilings = pilings_names[:3]
                urls_kislots = data[data['category'] =='кислота']['page_url'][:3]
                urls_pilings = data[data['category'] =='пилинг']['page_url'][:3]

                for name, url in zip(top_3_kislots, urls_kislots):
                    url = 'https://'+url
                    # Отправляем HTTP запрос и получаем ответ
                    await asyncio.sleep(1)
                    r = requests.get(url, headers=headers)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    drug_descs = soup.find('div', 'kr_review_plain_text').text
                    prompt_review  = f'{drug_descs[:4000]} \n\nВыдели главную мысль из всех отзывов. Ответ начни со слов: "Этот препарат...'
                    response_text = generate(prompt_review)
                    kislots_reviews[name] = response_text
                
                # Вывод результатов для кислот
                smiley = category_smileys.get('кислота', "😃")
                result = f'\n\n{smiley} Кислоты:'
                for i, (name, reviews) in enumerate(kislots_reviews.items(), start=1):
                    result += f'\n\n{i}. {name}. \n\n{reviews}'
                await bot.send_message(user_id, result)
                await bot.send_message(user_id, '\n\nДумаю... ⏳')

                for name, url in zip(top_3_pilings, urls_pilings):
                    url = 'https://'+url
                    # Отправляем HTTP запрос и получаем ответ
                    await asyncio.sleep(1)
                    r = requests.get(url, headers=headers)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    drug_descs = soup.find('div', 'kr_review_plain_text').text
                    prompt_review  = f'{drug_descs[:4000]} \n\nВыдели главную мысль из всех отзывов. Ответ начни со слов: "Этот препарат...'
                    response_text = generate(prompt_review)
                    pilings_reviews[name] = response_text

                # Вывод результатов для пилингов
                smiley = category_smileys.get('пилинг', "😃")
                result = f'\n\n{smiley} Пилинги:'
                for i, (name, reviews) in enumerate(pilings_reviews.items(), start=1):
                    result += f'\n\n{i}. {name}. \n\n{reviews}'
                await bot.send_message(user_id, result)

            except Exception as ex:
                result = f'\nОшибка 2: {ex}'
                await bot.send_message(user_id, result)


    elif len(detected_labels)>=2:
        detected_labels = ['postacne' if x in ['papula', 'pustula'] else x for x in detected_labels]
        for problem in detected_labels:
            if problem != 'hyperceratos':
                if problem in ['papula', 'postacne', 'pustula']:
                    data = pd.read_csv("datasets/clean/acne_clean_with_categories.csv")
                elif problem == "cuperos":
                    data = pd.read_csv("datasets/clean/cuperoz_clean_with_categories.csv")
                elif problem == 'camedon':
                    data = pd.read_csv("datasets/clean/camedons_clean_with_categories.csv")

                try:
                    all_drug_responses = {}
                    for x in data['category'].unique():
                        result = ''
                        drug_responses = {}
                        category_data = data[data['category'] == x]
                        urls = category_data['page_url'][:3]
                        top_3_names = category_data['drug_name'][:3].tolist()

                        for name, url in zip(top_3_names, urls):
                            url = 'https://'+url
                        # Отправляем HTTP запрос и получаем ответ
                            await asyncio.sleep(1)
                            r = requests.get(url, headers=headers)
                            soup = BeautifulSoup(r.text, 'html.parser')
                            drug_descs = soup.find('div', 'kr_review_plain_text').text
                            prompt_review  = f'{drug_descs[:4000]} \n\nВыдели главную мысль из всех отзывов. Ответ начни со слов: "Этот препарат...'
                            response_text = generate(prompt_review)
                            drug_responses[name] = response_text

                        all_drug_responses[x] = drug_responses    

                    for cat, drugnamesNreviews in all_drug_responses.items():
                        smiley = category_smileys.get(cat, "😃")  # По умолчанию используется "😃"
                        result = f"\n\n{smiley} Категория: {cat}"
                        for i, (name, drug_r3views) in enumerate(drugnamesNreviews.items(), start=1):
                            result += f'\n\n{i}. {name}. \n\n{drug_r3views}'
                        await bot.send_message(user_id, result)  
                        # await message.reply('\n\nДумаю... ⏳')

                except Exception as ex:
                    result = f'\nОшибка при получении отзывов с сайта. Пожалуйста, попробуйте еще раз.'
                    await bot.send_message(user_id, result)
                    
            else:
                data = pd.read_csv('datasets/clean/kislots_and_pilings.csv')
                kislots_names = data[data['category'] =='кислота']['drug_name'].tolist()
                pilings_names = data[data['category'] =='пилинг']['drug_name'].tolist()

                try:
                    result = ''
                    pilings_reviews = {}
                    kislots_reviews = {}
                    top_3_kislots = kislots_names[:3]
                    top_3_pilings = pilings_names[:3]
                    urls_kislots = data[data['category'] =='кислота']['page_url'][:3]
                    urls_pilings = data[data['category'] =='пилинг']['page_url'][:3]

                    for name, url in zip(top_3_kislots, urls_kislots):
                        url = 'https://'+url
                        # Отправляем HTTP запрос и получаем ответ
                        await asyncio.sleep(1)
                        r = requests.get(url, headers=headers)
                        soup = BeautifulSoup(r.text, 'html.parser')
                        drug_descs = soup.find('div', 'kr_review_plain_text').text
                        prompt_review  = f'{drug_descs[:4000]} \n\nВыдели главную мысль из всех отзывов. Ответ начни со слов: "Этот препарат...'
                        response_text = generate(prompt_review)
                        kislots_reviews[name] = response_text
                    
                    # Вывод результатов для кислот
                    smiley = category_smileys.get('кислота', "😃")
                    result = f'\n\n{smiley} Кислоты:'
                    for i, (name, reviews) in enumerate(kislots_reviews.items(), start=1):
                        result += f'\n\n{i}. {name}. \n\n{reviews}'
                    await bot.send_message(user_id, result) 
                    await bot.send_message(user_id, '\n\nДумаю... ⏳')

                    for name, url in zip(top_3_pilings, urls_pilings):
                        url = 'https://'+url
                        # Отправляем HTTP запрос и получаем ответ
                        await asyncio.sleep(1)
                        r = requests.get(url, headers=headers)
                        soup = BeautifulSoup(r.text, 'html.parser')
                        drug_descs = soup.find('div', 'kr_review_plain_text').text
                        prompt_review  = f'{drug_descs[:4000]} \n\nВыдели главную мысль из всех отзывов. Ответ начни со слов: "Этот препарат...'
                        response_text = generate(prompt_review)
                        pilings_reviews[name] = response_text

                    # Вывод результатов для пилингов
                    smiley = category_smileys.get('пилинг', "😃")
                    result = f'\n\n{smiley} Пилинги:'
                    for i, (name, reviews) in enumerate(pilings_reviews.items(),start=1):
                        result += f'\n\n{i}. {name}. \n\n{reviews}'
                    await bot.send_message(user_id, result) 

                except Exception as ex:
                    result = f'\nОшибка 2: {ex}'
                    await bot.send_message(user_id, result)
                
if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp) 
    
#  __name__  - это специальная переменная в Python, которая содержит имя текущего модуля. 
# Если модуль запущен напрямую, а не импортирован, то значение  __name__  будет равно  '__main__' .  
# executor.start_polling(dp) запускает бесконечный цикл опроса вх. сообщений для обработки,
# который ожидает новые сообщения от пользователей и передает их в соответствующие обработчики
