import os
import logging
import re
import tempfile
import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters
)
from PIL import Image
import google.generativeai as genai
from openai import OpenAI

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# #############################################################################
# ######## S E C U R I T Y   N O T I C E ######################################
#
# HARDCODING KEYS IS DANGEROUS.
# Anyone who sees this code can steal your keys and use your accounts.
# The best practice is to use Environment Variables like this:
#
# TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
#
# Then, set these variables in your hosting service's dashboard (e.g., Render).
#
# #############################################################################

# --- API Key Configuration (REPLACE WITH YOUR REAL KEYS) ---
API_KEYS = {
    "GEMINI_ANSWER": "AIzaSyANSjpUDyG-ekfcxuaaDlBjJ1SN1jhXkrM",  # Replace with your Gemini API Key
    "GEMINI_TEXT": "AIzaSyB8801ZIw0hLj6SvTAQ5Fs7Dw9C0711j0U",    # Replace with your Gemini API Key (can be the same)
    "OPENROUTER": "sk-or-v1-a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", # Replace with your OpenRouter API Key
}

# --- Telegram Bot Token (REPLACE WITH YOUR REAL TOKEN) ---
TELEGRAM_BOT_TOKEN = "7987505867:AAHSkgPQixRAivY357LaJiuIVRk2pP0Tb54" # Replace with your Telegram Bot Token

# --- Optional OpenRouter Headers ---
# You can customize these if you want your bot to appear on the OpenRouter leaderboard.
OPENROUTER_REFERER = "https://your-bot-url.com" # Optional: Your website or bot's URL
OPENROUTER_TITLE = "AI MCQ Solver Bot"         # Optional: Your project's name


# Validate that essential configurations are set
if "YOUR_" in TELEGRAM_BOT_TOKEN or len(TELEGRAM_BOT_TOKEN) < 20:
    logger.critical("CRITICAL: Your TELEGRAM_BOT_TOKEN is a placeholder or invalid. The bot cannot start.")
    exit()

# Initialize a single client for OpenRouter
openrouter_client = None
if API_KEYS["OPENROUTER"] and "sk-or-v1" in API_KEYS["OPENROUTER"]:
    try:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEYS["OPENROUTER"],
        )
        logger.info("OpenRouter client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter client: {e}")
else:
    logger.warning("OpenRouter client not initialized. Check your OPENROUTER API key.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /start command."""
    welcome_text = (
        "🤖 Welcome to AI MCQ Solver Bot!\n\n"
        "Send me a photo OR text of your question and I'll analyze it with multiple AI models!\n\n"
        "Supported formats: JPEG, PNG\n"
        "Max size: 5MB"
    )
    await update.message.reply_text(welcome_text)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles image uploads, extracts text, and gets answers from AI models."""
    message = update.message
    photo = message.photo[-1] if message.photo else None
    
    if not photo:
        await message.reply_text("Please send a proper image file (JPEG or PNG).")
        return
    if photo.file_size > 5 * 1024 * 1024: # 5MB limit
        await message.reply_text("Image size exceeds 5MB. Please send a smaller image.")
        return

    img_path = None
    status_msg = None
    try:
        file_info = await context.bot.get_file(photo.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
            await file_info.download_to_memory(temp_image_file)
            img_path = temp_image_file.name

        status_msg = await message.reply_text("🔍 Analyzing image and preparing for AI models...")
        
        # Task 1: Get a direct answer from Gemini Vision
        gemini_vision_task = asyncio.create_task(gemini_answer_from_image(img_path))
        
        # Task 2: Extract text from the image using Gemini
        extracted_text = await gemini_text_extract(img_path)
        
        if extracted_text and extracted_text.strip():
            display_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
            escaped_text = re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", display_text)
            await message.reply_text(f"📝 Extracted Text:\n```\n{escaped_text}\n```", parse_mode="MarkdownV2")
        else:
            await message.reply_text("ℹ️ No usable text was extracted from the image.")

        # Process the Gemini Vision result first
        try:
            model_name, result = await gemini_vision_task
            await message.reply_text(f"✦ {model_name}:\n{result}")
        except Exception as e:
            logger.error(f"Error processing Gemini Vision (image) task result: {e}", exc_info=True)
            await message.reply_text(f"✦ Gemini Vision (Image):\nError processing result: {str(e)}")

        # If text was extracted, process it with the other models
        if extracted_text and extracted_text.strip():
            await message.reply_text("🧠 Processing extracted text with other models via OpenRouter...")
            await process_text_with_models(message, extracted_text)
        
        if status_msg: await status_msg.delete()
        await message.reply_text("✅ Image processing complete.")

    except Exception as e:
        logger.error(f"Overall image processing error: {e}", exc_info=True)
        await message.reply_text(f"❌ A critical error occurred while processing the image: {str(e)}")
        if status_msg:
            try: await status_msg.delete()
            except Exception: pass
    finally:
        if img_path and os.path.exists(img_path):
            try: os.unlink(img_path)
            except Exception as e: logger.error(f"Error deleting temp file {img_path}: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles direct text messages and gets answers from all configured AI models."""
    text = update.message.text
    if not text or text.isspace():
        await update.message.reply_text("Please provide some text for the question.")
        return

    status_msg = await update.message.reply_text("🔍 Processing your text question with all AI models...")
    try:
        # Create a task for Gemini
        gemini_task = asyncio.create_task(gemini_answer_from_text(text))
        
        # Process with OpenRouter models in parallel
        await process_text_with_models(update.message, text)
        
        # Wait for and process the Gemini result
        try:
            model_name, result = await gemini_task
            await update.message.reply_text(f"✦ {model_name}:\n{result}")
        except Exception as e:
            logger.error(f"Error processing Gemini (text) task result: {e}", exc_info=True)

        if status_msg: await status_msg.delete()
        await update.message.reply_text("✅ Text question processing complete.")
    except Exception as e:
        logger.error(f"Overall text processing error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ A critical error occurred while processing the text: {str(e)}")
        if status_msg:
            try: await status_msg.delete()
            except Exception: pass

async def process_text_with_models(message, text):
    """Helper function to process a piece of text with all configured OpenRouter models."""
    tasks = []
    if openrouter_client:
        tasks.append(process_openrouter_model("Anthropic Sonnet 4.5", text, "anthropic/claude-sonnet-4.5"))
        tasks.append(process_openrouter_model("Grok 4 Fast", text, "x-ai/grok-4-fast"))
        tasks.append(process_openrouter_model("Deepseek Chat V3", text, "deepseek/deepseek-chat-v3-0324"))
        tasks.append(process_openrouter_model("OpenAI GPT-5", text, "openai/gpt-5"))
    else:
        await message.reply_text("ℹ️ OpenRouter client is not configured, skipping models.")
        return

    # Await and send results as they complete
    for future in asyncio.as_completed(tasks):
        try:
            model_name, result = await future
            await message.reply_text(f"✦ {model_name}:\n{result}")
        except Exception as e:
            logger.error(f"Error processing a model's future: {e}", exc_info=True)

async def gemini_answer_from_image(img_path):
    """Gets an answer directly from an image using Gemini Pro Vision."""
    if not API_KEYS["GEMINI_ANSWER"]:
        return ("Gemini Vision (Image)", "Error: API Key is not configured.")
    try:
        genai.configure(api_key=API_KEYS["GEMINI_ANSWER"])
        img = Image.open(img_path)
        model = genai.GenerativeModel("gemini-2.5-pro")
        prompt = "For multiple-choice questions (MCQs), respond only with the correct option in the format: 'The correct answer is: a) [option]'. Do not provide any explanation. For non-MCQ questions, provide a short, clear, and accurate answer without unnecessary detail."
        response = await model.generate_content_async([prompt, img])
        return ("Gemini Vision (Image)", clean_response(response.text))
    except Exception as e:
        logger.error(f"Gemini Vision (image answer) API error: {e}", exc_info=True)
        return ("Gemini Vision (Image)", f"API Error: {str(e)}")

async def gemini_answer_from_text(text):
    """Gets an answer from a text string using Gemini Pro."""
    if not API_KEYS["GEMINI_ANSWER"]:
        return ("Gemini (Text)", "Error: API Key is not configured.")
    try:
        genai.configure(api_key=API_KEYS["GEMINI_ANSWER"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = "For multiple-choice questions (MCQs), respond only with the correct option in the format: 'The correct answer is: a) [option]'. Do not provide any explanation. For non-MCQ questions, provide a short, clear, and accurate answer without unnecessary detail."
        response = await model.generate_content_async([prompt, text])
        return ("Gemini (Text)", clean_response(response.text))
    except Exception as e:
        logger.error(f"Gemini Text (answer) API error: {e}", exc_info=True)
        return ("Gemini (Text)", f"API Error: {str(e)}")

async def process_openrouter_model(model_name_display, query_text, openrouter_model_id):
    """Processes a query with a specific model via the OpenRouter client."""
    system_prompt = (
        "For multiple-choice questions (MCQs), respond only with the correct option "
        "in the format: 'The correct answer is: a) [option]'. Do not provide any explanation. "
        "For non-MCQ questions, provide a short, clear, and accurate answer without "
        "unnecessary detail."
    )
    
    try:
        completion = await asyncio.to_thread(
            openrouter_client.chat.completions.create,
            model=openrouter_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_text}
            ],
            extra_headers={
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_TITLE,
            },
            temperature=0.1
        )
        response_content = completion.choices[0].message.content
        return (model_name_display, clean_response(response_content))
    
    except Exception as e:
        logger.error(f"Error processing model {model_name_display} ({openrouter_model_id}) via OpenRouter: {e}", exc_info=True)
        return (model_name_display, f"API Error: {str(e)}")

async def gemini_text_extract(img_path):
    """Extracts text from an image using Gemini Pro Vision."""
    if not API_KEYS["GEMINI_TEXT"]:
        logger.warning("Cannot extract text, GEMINI_TEXT API key is missing.")
        return None
    try:
        genai.configure(api_key=API_KEYS["GEMINI_TEXT"])
        img = Image.open(img_path)
        model = genai.GenerativeModel("gemini-2.5-pro")
        prompt = "Extract all text exactly as it appears in the image. Preserve line breaks and formatting as much as possible. Output only the extracted text."
        response = await model.generate_content_async([prompt, img])
        extracted_text = response.text.strip()
        # Clean up markdown code blocks if the model adds them
        if extracted_text.startswith("```") and extracted_text.endswith("```"):
            extracted_text = re.sub(r"^```[a-zA-Z]*\n?", "", extracted_text)
            extracted_text = re.sub(r"\n?```$", "", extracted_text)
        logger.info(f"Gemini Text Extraction (first 100 chars): {extracted_text[:100]}")
        return extracted_text.strip() if extracted_text else None
    except Exception as e:
        logger.error(f"Gemini Text Extraction API Failed: {e}", exc_info=True)
        return None

def clean_response(text):
    """Cleans the AI model's response to be concise and well-formatted."""
    if not text: return ""
    # Remove thought process tags if they exist
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Try to find a clear MCQ answer format first
    mcq_patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\-]*\s*(?:option\s+)?([a-zA-Z0-9]+[.)])\s*([^<\n]+?)(?:\s*<|$|\n)",
        r"^\s*([a-zA-Z0-9]+[.)])\s+(.+)$", # e.g., "a) The answer"
    ]
    for pattern in mcq_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
        if match:
            option_marker = match.group(1).strip()
            option_text = match.group(2).strip('.,:;\"\' ') if len(match.groups()) > 1 and match.group(2) else ""
            return f"{option_marker} {option_text}".strip()

    # If no specific MCQ format, return the first non-empty line if it's short
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line.split()) <= 10 and len(first_line) < 100:
            return first_line
        # Otherwise, return a truncated version of the first line
        return first_line[:150] + "..." if len(first_line) > 150 else first_line
        
    return cleaned # Fallback to the cleaned text if nothing else matches

def main():
    """Starts the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN is not set. Bot cannot start.")
        return
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    logger.info("Bot is starting to poll...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
