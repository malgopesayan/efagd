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
from openai import OpenAI # Still used for NVIDIA
import groq

# Azure SDK for GitHub Models
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Passkey Configuration ---
PASSKEY = "86700"
AUTHORIZED_USERS = set()


# --- API Key Configuration ---
API_KEYS = {
    "GEMINI_ANSWER": os.getenv("GEMINI_ANSWER"),
    "GEMINI_TEXT": os.getenv("GEMINI_TEXT"),
    "NVIDIA": os.getenv("NVIDIA"),
    "GROQ": os.getenv("GROQ"),
    "GITHUB_GPT": os.getenv("GITHUB_GPT_PAT"),
    "GITHUB_GROK": os.getenv("GITHUB_GROK_PAT"),
}

# --- Telegram Bot Token ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7987505867:AAHSkgPQixRAivY357LaJiuIVRk2pP0Tb54")

# Validate configurations
essential_services = {
    "Telegram Bot Token": TELEGRAM_BOT_TOKEN,
    "Gemini Answer API Key": API_KEYS["GEMINI_ANSWER"],
    "Gemini Text API Key": API_KEYS["GEMINI_TEXT"],
    "Nvidia API Key": API_KEYS["NVIDIA"],
    "Groq API Key": API_KEYS["GROQ"],
    "GitHub PAT for GPT Models": API_KEYS["GITHUB_GPT"],
    "GitHub PAT for Grok 3 Model": API_KEYS["GITHUB_GROK"],
}
missing_or_placeholder_configs = [
    name for name, value in essential_services.items()
    if not value or "YOUR_" in str(value).upper() or "_HERE" in str(value).upper() or "PLACEHOLDER" in str(value).upper()
]
if missing_or_placeholder_configs:
    logger.critical(
        f"CRITICAL: The following configurations are missing or placeholders: "
        f"{', '.join(missing_or_placeholder_configs)}. Bot may not work as expected."
    )

# Initialize AI clients
groq_client = None
nvidia_client = None
github_gpt_client_azure_sdk = None # Client for GPT models via GitHub
github_grok_client_azure_sdk = None # Client for Grok 3 model via GitHub

try:
    if API_KEYS["GROQ"] and "YOUR_" not in API_KEYS["GROQ"].upper() and "PLACEHOLDER" not in API_KEYS["GROQ"].upper():
        groq_client = groq.Client(api_key=API_KEYS["GROQ"])
        logger.info("Groq client initialized.")
    else:
        logger.warning("Groq client not initialized (API key for GROQ missing/placeholder).")

    if API_KEYS["NVIDIA"] and "YOUR_" not in API_KEYS["NVIDIA"].upper() and "PLACEHOLDER" not in API_KEYS["NVIDIA"].upper():
        nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=API_KEYS["NVIDIA"]
        )
        logger.info("Nvidia client initialized.")
    else:
        logger.warning("Nvidia client not initialized (API key for NVIDIA missing/placeholder).")

    # Initialize GitHub Models client for GPT Models
    if API_KEYS["GITHUB_GPT"] and "YOUR_" not in API_KEYS["GITHUB_GPT"].upper() and "PLACEHOLDER" not in API_KEYS["GITHUB_GPT"].upper():
        try:
            github_gpt_client_azure_sdk = ChatCompletionsClient(
                endpoint="https://models.github.ai/inference",
                credential=AzureKeyCredential(API_KEYS["GITHUB_GPT"])
            )
            logger.info("GitHub Models client (Azure SDK for GPT models) initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GitHub Models client for GPT (Azure SDK): {e}")
    else:
        logger.warning("GitHub Models client (Azure SDK for GPT models) not initialized (GITHUB_GPT_PAT missing/placeholder).")

    # Initialize GitHub Models client for Grok 3 Model
    if API_KEYS["GITHUB_GROK"] and "YOUR_" not in API_KEYS["GITHUB_GROK"].upper() and "PLACEHOLDER" not in API_KEYS["GITHUB_GROK"].upper():
        try:
            github_grok_client_azure_sdk = ChatCompletionsClient(
                endpoint="https://models.github.ai/inference",
                credential=AzureKeyCredential(API_KEYS["GITHUB_GROK"])
            )
            logger.info("GitHub Models client (Azure SDK for Grok 3 model) initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GitHub Models client for Grok 3 (Azure SDK): {e}")
    else:
        logger.warning("GitHub Models client (Azure SDK for Grok 3 model) not initialized (GITHUB_GROK_PAT missing/placeholder).")

except Exception as e:
    logger.error(f"Error during global AI client initializations: {e}")


# --- NEW: Login command handler ---
async def login(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /login command to authorize users."""
    try:
        provided_key = context.args[0]
        if provided_key == PASSKEY:
            AUTHORIZED_USERS.add(update.message.from_user.id)
            await update.message.reply_text("✅ Access granted. You can now use the bot.")
            logger.info(f"User {update.message.from_user.id} authenticated successfully.")
        else:
            await update.message.reply_text("❌ Access denied. Incorrect passkey.")
            logger.warning(f"Failed login attempt by user {update.message.from_user.id}.")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /login <passkey>")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🤖 Welcome to AI MCQ Solver Bot!\n\n"
        "This bot is protected. Please use the /login command with the correct passkey to begin.\n\n"
        "Example: `/login 12345`"
    )
    await update.message.reply_text(welcome_text)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # --- MODIFIED: Authorization Check ---
    if update.message.from_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("🚫 Unauthorized. Please use `/login <passkey>` to get access.")
        return

    message = update.message
    photo = message.photo[-1] if message.photo else None
    
    if not photo:
        await message.reply_text("Please send a proper image file (JPEG or PNG).")
        return
    if photo.file_size > 5 * 1024 * 1024: # 5MB
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
        
        gemini_vision_task = None
        if API_KEYS["GEMINI_ANSWER"] and "YOUR_" not in API_KEYS["GEMINI_ANSWER"].upper():
            gemini_vision_task = asyncio.create_task(gemini_answer_from_image(img_path))
            logger.info("Gemini Vision (Image Answer) task added.")
        else:
            logger.warning("Gemini Vision (Image Answer) task NOT added: API_KEY['GEMINI_ANSWER'] missing or placeholder.")
        
        extracted_text = None
        if API_KEYS["GEMINI_TEXT"] and "YOUR_" not in API_KEYS["GEMINI_TEXT"].upper():
            try:
                extracted_text = await gemini_text_extract(img_path)
                if extracted_text and extracted_text.strip():
                    display_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                    escaped_text = re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", display_text)
                    await message.reply_text(f"📝 Extracted Text:\n```\n{escaped_text}\n```", parse_mode="MarkdownV2")
                else:
                    await message.reply_text("ℹ️ No text extracted by Gemini or text is empty.")
            except Exception as e:
                logger.error(f"Error in Gemini Text Extraction: {e}", exc_info=True)
                await message.reply_text(f"✦ Gemini Text Extraction:\nError: {str(e)}")
        else:
            logger.warning("Gemini Text Extraction task NOT added: API_KEY['GEMINI_TEXT'] missing or placeholder.")
        
        if gemini_vision_task:
            try:
                model_name, result = await gemini_vision_task
                await message.reply_text(f"✦ {model_name}:\n{result}")
            except Exception as e:
                logger.error(f"Error processing Gemini Vision (image) task result: {e}", exc_info=True)
                await message.reply_text(f"✦ Gemini Vision (Image):\nError processing result: {str(e)}")

        if extracted_text and extracted_text.strip():
            await message.reply_text("🧠 Processing extracted text with other models...")
            text_model_processing_tasks = []
            if nvidia_client:
                text_model_processing_tasks.append(process_model_with_name("NVIDIA", extracted_text, "nvidia"))
            if groq_client:
                text_model_processing_tasks.append(process_model_with_name("Llama (Groq)", extracted_text, "llama"))
                text_model_processing_tasks.append(process_model_with_name("Deepseek (Groq)", extracted_text, "deepseek"))
            
            # Use specific client for GPT models
            if github_gpt_client_azure_sdk:
                text_model_processing_tasks.append(process_model_with_name("GPT-4o (GitHub)", extracted_text, "gpt4o_github_azure_sdk"))
            
            # Use specific client for Grok 3 model
            if github_grok_client_azure_sdk:
                text_model_processing_tasks.append(process_model_with_name("Grok 3 (GitHub)", extracted_text, "grok_github_azure_sdk"))

            if text_model_processing_tasks:
                for future in asyncio.as_completed(text_model_processing_tasks):
                    try:
                        model_name, result = await future
                        await message.reply_text(f"✦ {model_name}:\n{result}")
                    except Exception as e:
                        logger.error(f"Error processing a text model's future for image's extracted text: {e}", exc_info=True)
            else:
                await message.reply_text("ℹ️ No additional text-based AI models configured for extracted text.")
        elif not extracted_text and API_KEYS["GEMINI_TEXT"]:
             await message.reply_text("ℹ️ Could not extract text from image to process with other models.")
        
        if status_msg: await status_msg.delete()
        await message.reply_text("✅ Image processing complete.")

    except Exception as e:
        logger.error(f"Overall image processing error: {e}", exc_info=True)
        await message.reply_text(f"❌ Critical error processing image: {str(e)}")
        if status_msg:
            try: await status_msg.delete()
            except Exception: pass
    finally:
        if img_path and os.path.exists(img_path):
            try: os.unlink(img_path)
            except Exception as e: logger.error(f"Error deleting temp file {img_path}: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # --- MODIFIED: Authorization Check ---
    if update.message.from_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("🚫 Unauthorized. Please use `/login <passkey>` to get access.")
        return
        
    text = update.message.text
    if not text or text.isspace():
        await update.message.reply_text("Please provide text for the question.")
        return

    status_msg = await update.message.reply_text("🔍 Processing your text question...")
    try:
        tasks = []
        if API_KEYS["GEMINI_ANSWER"] and "YOUR_" not in API_KEYS["GEMINI_ANSWER"].upper():
            tasks.append(gemini_answer_from_text(text))
            logger.info("Gemini (Text Answer) task added.")
        else:
            logger.warning("Gemini (Text Answer) task NOT added: API_KEY['GEMINI_ANSWER'] missing or placeholder.")

        # if nvidia_client:
        #     tasks.append(process_model_with_name("NVIDIA", text, "nvidia"))
        if groq_client:
            tasks.append(process_model_with_name("Llama (Groq)", text, "llama"))
            tasks.append(process_model_with_name("Deepseek (Groq)", text, "deepseek"))
        
        # Use specific client for GPT models
        if github_gpt_client_azure_sdk:
            tasks.append(process_model_with_name("GPT-4o (GitHub)", text, "gpt4o_github_azure_sdk"))
        
        # Use specific client for Grok 3 model
        # if github_grok_client_azure_sdk:
        #     tasks.append(process_model_with_name("Grok 3 (GitHub)", text, "grok_github_azure_sdk"))

        if not tasks:
            await update.message.reply_text("ℹ️ No AI models are configured or active for text processing.")
            if status_msg: await status_msg.delete()
            return

        for future in asyncio.as_completed(tasks):
            try:
                model_name, result = await future
                await update.message.reply_text(f"✦ {model_name}:\n{result}")
            except Exception as e:
                logger.error(f"Error processing a text model's future for text input: {e}", exc_info=True)
        
        if status_msg: await status_msg.delete()
        await update.message.reply_text("✅ Text question processing complete.")
    except Exception as e:
        logger.error(f"Overall text processing error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Critical error processing text: {str(e)}")
        if status_msg:
            try: await status_msg.delete()
            except Exception: pass

async def gemini_answer_from_image(img_path):
    try:
        genai.configure(api_key=API_KEYS["GEMINI_ANSWER"])
        img = Image.open(img_path)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt_parts = [
            "For multiple-choice questions (MCQs), respond only with the correct option in the format: 'The correct answer is: a) [option]'. Do not provide any explanation. For non-MCQ questions, provide a short, clear, and accurate answer without unnecessary detail.",
            img
        ]
        response = await model.generate_content_async(prompt_parts)
        return ("Gemini Vision (Image)", clean_response(response.text))
    except Exception as e:
        logger.error(f"Gemini Vision (image answer) API error: {e}", exc_info=True)
        return ("Gemini Vision (Image)", f"API Error: {str(e)}")

async def gemini_answer_from_text(text):
    try:
        genai.configure(api_key=API_KEYS["GEMINI_ANSWER"])
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt_parts = [
            "For multiple-choice questions (MCQs), respond only with the correct option in the format: 'The correct answer is: a) [option]'. Do not provide any explanation. For non-MCQ questions, provide a short, clear, and accurate answer without unnecessary detail.",
            text
        ]
        response = await model.generate_content_async(prompt_parts)
        return ("Gemini (Text)", clean_response(response.text))
    except Exception as e:
        logger.error(f"Gemini Text (answer) API error: {e}", exc_info=True)
        return ("Gemini (Text)", f"API Error: {str(e)}")

async def process_model_with_name(model_name_display, query_text, model_key_internal):
    try:
        system_prompt_content = (
            "For multiple-choice questions (MCQs), respond only with the correct option "
            "in the format: 'The correct answer is: a) [option]'. Do not provide any explanation. "
            "For non-MCQ questions, provide a short, clear, and accurate answer without "
            "unnecessary detail."
        )
        user_content = query_text
        response_content = ""

        # if model_key_internal == "nvidia" and nvidia_client:
        #     messages_payload=[{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_content}]
        #     completion = await asyncio.to_thread(
        #         nvidia_client.chat.completions.create,
        #         model="nvidia/nemotron-4-340b-instruct", messages=messages_payload, temperature=0.1
        #     )
        #     response_content = completion.choices[0].message.content
        if model_key_internal == "llama" and groq_client:
            messages_payload=[{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_content}]
            completion = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model="llama3-70b-8192", messages=messages_payload, temperature=0.1
            )
            response_content = completion.choices[0].message.content
        elif model_key_internal == "deepseek" and groq_client:
            target_deepseek_model = "deepseek-coder-33b-instruct" # Example model, adjust if needed
            messages_payload=[{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_content}]
            completion = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=target_deepseek_model, messages=messages_payload, temperature=0.1
            )
            response_content = completion.choices[0].message.content
        elif model_key_internal == "gpt4o_github_azure_sdk" and github_gpt_client_azure_sdk: # Use GPT client
            azure_sdk_messages = [SystemMessage(content=system_prompt_content), UserMessage(content=user_content)]
            completion = await asyncio.to_thread(
                github_gpt_client_azure_sdk.complete, model="gpt-4o", messages=azure_sdk_messages, temperature=0.1
            )
            response_content = completion.choices[0].message.content
        elif model_key_internal == "grok_github_azure_sdk" and github_grok_client_azure_sdk: # Use Grok client
            azure_sdk_messages = [SystemMessage(content=system_prompt_content), UserMessage(content=user_content)]
            completion = await asyncio.to_thread(
                github_grok_client_azure_sdk.complete, model="grok-1", messages=azure_sdk_messages, temperature=0.1
            )
            response_content = completion.choices[0].message.content
        else:
            client_name = "Unknown"
            if model_key_internal == "gpt4o_github_azure_sdk": client_name = "GitHub GPT Client"
            elif model_key_internal == "grok_github_azure_sdk": client_name = "GitHub Grok Client"
            logger.error(f"{client_name} not available or unknown model_key: {model_key_internal} for {model_name_display}")
            return (model_name_display, f"Error: {client_name} not available or config issue.")

        return (model_name_display, clean_response(response_content))
    
    except HttpResponseError as e_azure:
        logger.error(f"Azure SDK HttpResponseError for {model_name_display} ({model_key_internal}): {e_azure.status_code} - {e_azure.message}", exc_info=True)
        error_detail = e_azure.message
        try:
            error_body = e_azure.response.json()
            if error_body and 'error' in error_body and 'message' in error_body['error']:
                error_detail = error_body['error']['message']
        except Exception: pass
        return (model_name_display, f"API Error: {e_azure.status_code} - {e_azure.reason} ({error_detail})")
    except groq.APIError as e_groq:
        logger.error(f"Groq APIError for {model_name_display} ({model_key_internal}): {e_groq}", exc_info=True)
        error_message = str(e_groq)
        if hasattr(e_groq, 'body') and e_groq.body and 'error' in e_groq.body and 'message' in e_groq.body['error']:
            error_message = e_groq.body['error']['message']
        return (model_name_display, f"API Error (Groq): {error_message}")
    except Exception as e:
        logger.error(f"Error processing model {model_name_display} ({model_key_internal}): {e}", exc_info=True)
        return (model_name_display, f"API Error: {str(e)}")

async def gemini_text_extract(img_path):
    try:
        genai.configure(api_key=API_KEYS["GEMINI_TEXT"])
        img = Image.open(img_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_parts = [
            "Extract all text exactly as it appears in the image. Preserve line breaks and formatting as much as possible. Output only the extracted text.",
            img
        ]
        response = await model.generate_content_async(prompt_parts)
        extracted_text = response.text.strip()
        if extracted_text.startswith("```") and extracted_text.endswith("```"):
            extracted_text = re.sub(r"^```[a-zA-Z]*\n?", "", extracted_text)
            extracted_text = re.sub(r"\n?```$", "", extracted_text)
        logger.info(f"Gemini Text Extraction (first 100 chars): {extracted_text[:100]}")
        return extracted_text.strip() if extracted_text else None
    except Exception as e:
        logger.error(f"Gemini Text Extraction API Failed: {e}", exc_info=True)
        return None

def clean_response(text):
    if not text: return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()    
    mcq_patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\-]*\s*(?:option\s+)?([a-zA-Z0-9]+[.)])\s*([^<\n]+?)(?:\s*<|$|\n)",
        r"(?:the\s+)?correct\s+option\s+is\s*[:\-]*\s*([a-zA-Z0-9]+[.)])\s*([^<\n]*?)(?:\s*<|$|\n)",
        r"^\s*([a-zA-Z0-9]+[.)])\s+(.+)$",
        r"^\s*([a-zA-Z0-9](?:[.)])?)\s*$", 
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\-]*\s*(?:option\s+)?([a-zA-Z0-9](?![\w\s]*[):]))\s+([^<\n]+?)?(?:\s*<|$|\n)",
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\-]*\s*(?:option\s+)?([A-Z0-9])(?:\s*<|$|\n)"
    ]
    for pattern in mcq_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
        if match:
            groups = match.groups()
            option_marker = groups[0].strip()
            option_text = groups[1].strip() if len(groups) > 1 and groups[1] else ""
            
            if len(option_marker) == 1 and option_marker.isalnum() and not option_marker[-1] in '.)':
                 option_marker += ")" 
            
            if option_text: return f"{option_marker} {option_text.splitlines()[0].strip('.,:;\"\' ')}".strip()
            else: return f"{option_marker}".strip()

    general_match = re.search(r"(?:Correct Answer|Answer|The answer is|Solution)\s*[:\-]*\s*(.*?)(?:$|\n)", cleaned, re.IGNORECASE | re.MULTILINE)
    if general_match and general_match.group(1).strip(): return general_match.group(1).strip().splitlines()[0].strip('.,:;\"\' ')
    
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line.split()) <= 7 and len(first_line) < 70: return first_line
        if len(first_line) <= 3 and re.match(r"^[a-zA-Z0-9]+[.)]?$", first_line): return first_line
        return first_line[:150] + "..." if len(first_line) > 150 else first_line
        
    return cleaned


def main():
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN_HERE" in TELEGRAM_BOT_TOKEN or "PLACEHOLDER" in TELEGRAM_BOT_TOKEN.upper():
        logger.critical("TELEGRAM_BOT_TOKEN not set or placeholder. Bot cannot start.")
        return
    if missing_or_placeholder_configs:
         logger.warning(f"Bot starting with missing/placeholder configs for: {', '.join(missing_or_placeholder_configs)}. Functionality may be limited.")
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # --- MODIFIED: Add login handler first ---
    application.add_handler(CommandHandler("login", login))
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_image)) 
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Bot starting to poll...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
