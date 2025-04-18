from dotenv import load_dotenv
import os
 
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    filters,
    CallbackContext,
)
 
from alith import Agent, MilvusStore, chunk_text
 
# --------------------------------------------
# Constants
# --------------------------------------------

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_TELEGRAM_USER_ID = os.getenv("ADMIN_TELEGRAM_USER_ID")
 
# --------------------------------------------
# Alith Agent
# --------------------------------------------
 
agent = Agent(
    name="Telegram Bot Agent",
    model="gpt-4",
    preamble="""You are a spam detection bot. You will be given a message and your task is to determine if it is spam or not. If you think the message is spam, you will return \"true\". If you think the message is not spam, you will return \"false\".""",
    store=MilvusStore(),
)
 
# --------------------------------------------
# Telegram Bot
# --------------------------------------------

# Utilities

async def print_milvus_contents(update: Update, context: CallbackContext) -> None:
    results = agent.store.search(" ".join(context.args))
    await context.bot.send_message(chat_id=update.effective_chat.id, text=str(results))

def store_message_in_vector_store(message_text):
    """
    Embeds and stores a single message into the Milvus vector store for RAG retrieval.
    """
    chunks = chunk_text(message_text, overlap_percent=0.2)
    agent.store.save_docs(chunks)

async def process_message(message_text, context: CallbackContext) -> None:
    response = agent.prompt(message_text)
    if "true" in response.lower():
        store_message_in_vector_store(message_text)
        await context.bot.send_message(chat_id=ADMIN_TELEGRAM_USER_ID, text=f"\"{message_text}\" is spam and stored.")
    else:
        await context.bot.send_message(chat_id=ADMIN_TELEGRAM_USER_ID, text=f"\"{message_text}\" is not spam.")

# Handlers

async def handle_message(update: Update, context: CallbackContext) -> None:
    # Restrict to a specific Telegram user
    allowed_user_id = None
    try:
        allowed_user_id = int(ADMIN_TELEGRAM_USER_ID) if ADMIN_TELEGRAM_USER_ID else None
    except Exception:
        pass
    user_id = update.effective_user.id if update.effective_user else None
    if allowed_user_id is not None and user_id != allowed_user_id:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="You are not authorized to use this bot.")
        return

    message = update.message

    # Prefer caption if present (common for forwarded/channel messages), else text
    message_text = None
    if message.caption:
        message_text = message.caption
    elif message.text:
        message_text = message.text

    # Only process if there is text/caption
    if message_text:
        await store_message_in_vector_store(message_text)
        await context.bot.send_message(chat_id=ADMIN_TELEGRAM_USER_ID, text=f"\"{message_text}\" is stored.")

async def handle_group_message(update: Update, context: CallbackContext) -> None:
    message = update.message
    if message is None:
        return
    # Only process group messages
    if update.effective_chat.type not in ["group", "supergroup"]:
        return

    # Prefer caption if present (common for forwarded/channel messages), else text
    message_text = None
    if message.caption:
        message_text = message.caption
    elif message.text:
        message_text = message.text

    if not message_text:
        return

    # Search for the message in the store
    store = MilvusStore()
    results = store.search(message_text, 3, 0.8) # 3 results, 0.8 similarity
    if results:
        # If there is a match, forward to agent
        await process_message(message_text, context)


app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(CommandHandler("search", print_milvus_contents))
app.add_handler(MessageHandler((filters.TEXT | filters.FORWARDED) & filters.ChatType.GROUPS, handle_group_message))
app.add_handler(MessageHandler(filters.TEXT | filters.FORWARDED, handle_message))


if __name__ == "__main__":
    app.run_polling()