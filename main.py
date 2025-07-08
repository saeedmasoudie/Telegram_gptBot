import os
from openai import OpenAI
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
import io

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "your_TELEGRAM_TOKEN")

# --- Groq Client ---
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# --- Global State for User Settings and Conversation History ---
user_data = {}  # Stores {'user_id': {'llm_model': 'model_id', 'chat_history': []}}

# Define available Groq models for text generation
AVAILABLE_TEXT_MODELS = {
    "llama3-8b-8192": "LLaMA 3 8B (Groq)",
    "llama3-70b-8192": "LLaMA 3 70B (Groq)",
    "mixtral-8x7b-32768": "Mixtral 8x7B (Groq)",
    "gemma-7b-it": "Gemma 7B (Groq)",
}

DEFAULT_TEXT_MODEL = "llama3-8b-8192"  # Default to a faster, cheaper Groq model


# --- Utility Functions ---
def get_user_data(user_id):
    """Initializes user data if it doesn't exist and returns it."""
    if user_id not in user_data:
        user_data[user_id] = {
            "llm_model": DEFAULT_TEXT_MODEL,
            "chat_history": []
        }
    return user_data[user_id]


def add_message_to_history(user_id, role, content):
    """Adds a message to the user's chat history."""
    user_data = get_user_data(user_id)
    user_data["chat_history"].append({"role": role, "content": content})
    # Optional: Keep history to a reasonable length to avoid exceeding token limits
    # and to manage memory. Adjust as needed.
    MAX_HISTORY_LENGTH = 10  # Keep last 10 messages (5 user, 5 assistant)
    if len(user_data["chat_history"]) > MAX_HISTORY_LENGTH:
        user_data["chat_history"] = user_data["chat_history"][-MAX_HISTORY_LENGTH:]


# --- Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    user_id = update.effective_user.id
    get_user_data(user_id)  # Ensure user data is initialized

    await update.message.reply_text(
        f"👋 Welcome! I'm a chatbot powered by Groq's lightning-fast AI.\n"
        f"Currently, I'm using *{AVAILABLE_TEXT_MODELS[DEFAULT_TEXT_MODEL]}* for text generation.\n"
        f"Use /help to see available commands.",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays a list of available commands."""
    help_text = (
        "Here are the commands you can use:\n\n"
        "*/start* - Start the bot.\n"
        "*/help* - Show this help message.\n"
        "*/model* - Choose the language model for text generation.\n"
        "*/clear* - Clear the current conversation history.\n"
        "Send a *voice message* or *audio file* for transcription.\n"
        "\n"
        "Just send me a text message for a regular chat with the currently selected model."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def choose_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows the user to choose the LLM model for text generation."""
    keyboard = []
    for model_id, model_name in AVAILABLE_TEXT_MODELS.items():
        keyboard.append([InlineKeyboardButton(model_name, callback_data=f"set_text_model_{model_id}")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose your preferred language model for text chat:", reply_markup=reply_markup)


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clears the conversation history for the user."""
    user_id = update.effective_user.id
    user_data[user_id]["chat_history"] = []  # Clear history for the current user
    await update.message.reply_text("✅ Conversation history cleared.")


# --- Callback Query Handler ---

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles button presses from inline keyboards."""
    query = update.callback_query
    await query.answer()  # Acknowledge the callback query

    user_id = query.from_user.id
    callback_data = query.data
    user_settings = get_user_data(user_id)

    if callback_data.startswith("set_text_model_"):
        chosen_model_id = callback_data.replace("set_text_model_", "")
        if chosen_model_id in AVAILABLE_TEXT_MODELS:
            user_settings["llm_model"] = chosen_model_id
            await query.edit_message_text(
                f"Text generation model set to *{AVAILABLE_TEXT_MODELS[chosen_model_id]}*.",
                parse_mode="Markdown"
            )
            # Also clear history when model changes to avoid context issues
            user_settings["chat_history"] = []
            await query.message.reply_text("Conversation history cleared for new model.")
        else:
            await query.edit_message_text("Invalid model selection.")


# --- Message Handlers ---

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles general text messages for LLM interaction."""
    user_msg = update.message.text
    user_id = update.effective_user.id
    user_settings = get_user_data(user_id)
    selected_model = user_settings["llm_model"]
    chat_history = user_settings["chat_history"]

    # Send "Thinking..." message and get its Message object
    thinking_message = await update.message.reply_text("Thinking...")

    # Add user message to history
    add_message_to_history(user_id, "user", user_msg)

    reply = ""  # Initialize reply
    try:
        completion = groq_client.chat.completions.create(
            model=selected_model,
            messages=chat_history,  # Send the full conversation history
            temperature=0.7,  # Adjust creativity (0.0-1.0)
            max_tokens=2048,  # Max tokens for the response
        )
        reply = completion.choices[0].message.content
        add_message_to_history(user_id, "assistant", reply)  # Add assistant reply to history

    except Exception as e:
        reply = f"❌ Groq API Error ({selected_model}):\n{e}"
        print(f"Groq API Error: {e}")  # Log error for debugging

    # Edit the "Thinking..." message with the actual reply
    try:
        await thinking_message.edit_text(reply)
    except Exception as e:
        # Fallback: if editing fails (e.g., message too old, or Telegram API issues), send as new message
        await update.message.reply_text(reply)
        print(f"Failed to edit message: {e}. Sending as new message instead.")


async def transcribe_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles audio messages and transcribes them using Groq's Whisper."""
    if not update.message.voice and not update.message.audio:
        await update.message.reply_text("Please send a voice message or audio file for transcription.")
        return

    file_id = None
    if update.message.voice:
        file_id = update.message.voice.file_id
    elif update.message.audio:
        file_id = update.message.audio.file_id

    if not file_id:  # Should not happen if filters work, but for safety
        await update.message.reply_text("Could not find audio to transcribe.")
        return

    # Send "Transcribing..." message and get its Message object
    transcribing_message = await update.message.reply_text(
        "Transcribing audio using Groq Whisper... This might take a moment.")

    transcript_text = ""  # Initialize transcript text
    try:
        telegram_file = await context.bot.get_file(file_id)
        audio_bytes = io.BytesIO()
        await telegram_file.download_to_memory(audio_bytes)

        file_extension = ".ogg" if update.message.voice else ".mp3"
        audio_bytes.name = f"audio_file{file_extension}"
        audio_bytes.seek(0)

        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_bytes
        )
        transcript_text = f"📝 *Transcription:*\n{transcript.text}"
    except Exception as e:
        transcript_text = f"❌ Transcription error: {e}"
        print(f"Transcription error: {e}")

    # Edit the "Transcribing..." message with the transcription result
    try:
        await transcribing_message.edit_text(transcript_text, parse_mode="Markdown")
    except Exception as e:
        # Fallback if editing fails
        await update.message.reply_text(transcript_text, parse_mode="Markdown")
        print(f"Failed to edit transcription message: {e}. Sending as new message instead.")


# --- Main Bot Setup ---

def main():
    """Starts the bot."""
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Command Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("model", choose_model))
    app.add_handler(CommandHandler("clear", clear_history))

    # Callback Query Handler for inline buttons
    app.add_handler(CallbackQueryHandler(button_callback_handler))

    # Message Handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, transcribe_audio))

    print("Bot is polling with Groq-only functionality...")
    app.run_polling()


if __name__ == "__main__":
    main()