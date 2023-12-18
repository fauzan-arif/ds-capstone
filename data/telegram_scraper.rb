import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson.objectid import ObjectId
from dotenv import load_dotenv
from telethon.sync     import TelegramClient, events
from telethon.sessions import StringSession
from telethon          import functions, types

load_dotenv('../.env');

tele_api_id   = os.environ.get("TELEGRAM_APP_ID")
tele_api_hash = os.environ.get("TELEGRAM_APP_HASH")

mongo_host = os.environ.get("MONGO_HOST")
mongo_user = os.environ.get("MONGO_USER")
mongo_pass = os.environ.get("MONGO_PASS")
mongo_url  = f"mongodb+srv://{mongo_user}:{mongo_pass}@{mongo_host}/?retryWrites=true&w=majority"

def get_database(url, dbname, appname=None):
    try:
        # host and port default to 'localhost' and '27017'
        # appname will appear in logs
        client = MongoClient(url, appname=appname)
        # try connection to the database
        # The ping command is cheap and does not require auth.
        client.admin.command('ping')
        return client[dbname] 
    
    except ConnectionFailure:
        print("Error: MongoDB Server not available")


db     = get_database(mongo_url, 'capstone', 'Capstone')
client = TelegramClient('jam_session23', tele_api_id, tele_api_hash)

async with client:
  for channel in db.channels.find({}):
    async for message in client.iter_messages(channel, limit=200):
      await db.messages.insert_one({
        'channel': channel,
        'text': message.raw_text,
        'date': message.date
      })


client.disconnect()