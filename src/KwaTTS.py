import os
import sys
import asyncio
import discord
import random
import traceback
import re
import json
import uuid
import shutil
import time
from typing import List
from pathlib import Path
from discord.ext import commands
from discord import app_commands

# ----------------- CONFIG LOADING -----------------
def load_server_config():
    try:
        with open('./config/serverconfig.json', 'r') as f:
            config = json.load(f)
            return {
                "token": config["token"],
                "text_channel_id": config["text_channel_id"],
                "voice_channel_id": config["voice_channel_id"],
                "guild_id": config["guild_id"]
            }
    except FileNotFoundError:
        sys.exit("❌ Missing serverconfig.json")
    except KeyError as e:
        sys.exit(f"❌ Missing key in serverconfig.json: {e}")

# Load Edge TTS voices
with open('./src/edgettsvoices.json', 'r') as f:
    edge_voices_data = json.load(f)
edge_voices = list(edge_voices_data.keys())

# Load server configuration
try:
    config = load_server_config()
    TOKEN = config["token"]
    TEXT_CHANNEL_ID = config["text_channel_id"]
    VOICE_CHANNEL_ID = config["voice_channel_id"]
    GUILD_ID = config["guild_id"]
except Exception as e:
    sys.exit(f"❌ Config loading failed: {e}")



# ----------------------------------
# GLOBALS
# ----------------------------------
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
voice_client = None
tts_queue = []
QUEUE_LOCK = asyncio.Lock()
IS_PLAYING = False
ACRONYM_CACHE = None
ACRONYM_LAST_MODIFIED = 0


# ----------------------------------
# Change Voice Edge Command
# ----------------------------------
@bot.tree.command(name="changevoiceedge", description="Set your Edge TTS voice", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(voice="The Edge TTS voice to use")
async def changevoiceedge(interaction: discord.Interaction, voice: str):
    global edge_voices
    if voice not in edge_voices:
        await interaction.response.send_message(f"Invalid voice: {voice}", ephemeral=True)
        return
    await interaction.response.send_message(f"Edge voice set to {voice}", ephemeral=True)
    userconfig = load_user_config(str(interaction.user.id))
    userconfig["service"] = "edge"
    userconfig["selected_edge_voice"] = voice
    save_user_config(str(interaction.user.id), userconfig)

@changevoiceedge.autocomplete("voice")
async def edge_voice_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    global edge_voices
    return [
        app_commands.Choice(name=voice, value=voice)
        for voice in edge_voices
        if current.lower() in voice.lower()
    ][:25]

# ----------------------------------
# Universal Settings Command
# ----------------------------------
@bot.tree.command(name="usersettings", description="Adjust your TTS settings", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(volume="Output volume (0-100, default 100)")
async def usersettings(
    interaction: discord.Interaction,
    volume: app_commands.Range[int, 0, 100] = 100
):
    userconfig = load_user_config(str(interaction.user.id))
    userconfig["volume"] = volume
    save_user_config(str(interaction.user.id), userconfig)
    await interaction.response.send_message(
        f"🔊 Volume set to {volume}%",
        ephemeral=True
    )

# ----------------------------------
# toggle ignoreme
# ----------------------------------
@bot.tree.command(name="ignoremetoggle", description="toggles ignore my messages", guild=discord.Object(id=GUILD_ID))
async def ignoremetoggle(interaction: discord.Interaction):
    userconfig = load_user_config(str(interaction.user.id))
    try:
        newstate = not userconfig["ignoreme"]
    except:
        newstate = True
    await interaction.response.send_message(f"IgnoreMe set to: {newstate}", ephemeral=True)
    userconfig["ignoreme"] = newstate
    save_user_config(str(interaction.user.id), userconfig)


# ----------------------------------
# Edge settings command
# ----------------------------------
@bot.tree.command(name="edgesettings", description="Set your Edge TTS pitch and volume offsets", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(
    pitchoffset="Pitch offset for Edge TTS (-30 to 30, default 0)",
    volumeoffset="Volume offset for Edge TTS (-50 to 50, default 0)"
)
async def edgesettings(
    interaction: discord.Interaction,
    pitchoffset: app_commands.Range[int, -30, 30] = 0,
    volumeoffset: app_commands.Range[int, -50, 50] = 0
):
    userconfig = load_user_config(str(interaction.user.id))
    userconfig["edgepitch"] = pitchoffset
    userconfig["edgevolume"] = volumeoffset
    save_user_config(str(interaction.user.id), userconfig)
    await interaction.response.send_message(
        f"Edge TTS settings updated: Pitch offset = {pitchoffset}, Volume offset = {volumeoffset}",
        ephemeral=True
    )


# ----------------------------------
# Join Command
# ----------------------------------
@bot.tree.command(name="join", description="Requests bot to join voice", guild=discord.Object(id=GUILD_ID))
async def join(interaction: discord.Interaction):
    await interaction.response.send_message(f"Requested join...", ephemeral=True)
    try:
        channel = bot.get_channel(VOICE_CHANNEL_ID)
        global voice_client
        if channel:
            voice_client = await channel.connect()
            print(f"Connected to voice: {channel.name}")
        else:
            print("❌ Voice channel not found")
    except Exception as e:
        print(f"❌ Voice connection failed: {e}")

# ----------------------------------
# Leave Command
# ----------------------------------
@bot.tree.command(name="leave", description="Requests bot to leave voice", guild=discord.Object(id=GUILD_ID))
async def leave(interaction: discord.Interaction):
    await interaction.response.send_message(f"Requested leave...", ephemeral=True)
    voice_client = interaction.guild.voice_client
    if voice_client and voice_client.is_connected():
        await voice_client.disconnect()
        print(f"✅ Left voice channel: {voice_client.channel.name}")
    else:
        print("❌ Bot is not in a voice channel")

# ----------------------------------
# gtts Command
# ----------------------------------
@bot.tree.command(name="changevoicegtts", description="Switch to Google TTS with optional country TLD", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(tldendpoint="Country TLD for accent (e.g. 'co.uk' for British English)")
async def changevoicegtts(interaction: discord.Interaction, tldendpoint: str = "com"):
    userconfig = load_user_config(str(interaction.user.id))
    userconfig["service"] = "gtts"
    userconfig["gtts_tld"] = tldendpoint.lower()
    save_user_config(str(interaction.user.id), userconfig)
    await interaction.response.send_message(
        f"Switched to Google TTS with {tldendpoint} domain", 
        ephemeral=True
    )

    
def windows_escape(text):
    return text.replace('"', '""').replace('^', '^^').replace('&', '^&')

# ----------------------------------
# Ready Event
# ----------------------------------
@bot.event
async def on_ready():
    print(f'Bot ready: {bot.user}')
    try:
        guild = discord.Object(id=GUILD_ID)
        print(f"Commands in tree: {[cmd.name for cmd in bot.tree.get_commands(guild=guild)]}")
        synced = await bot.tree.sync(guild=guild)
        print(f"✅ Synced commands: {[cmd.name for cmd in synced]}")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")

async def generate_audio(task: dict) -> discord.FFmpegPCMAudio:
    start_time = time.time()
    safe_message = windows_escape(task["content"])
    output_file = task.get("debug_mp3") or task.get("debug_wav")
    service = task.get("service", "edge")

    try:
        if service == "gtts":
            command = build_gtts_command(task, safe_message)
        elif service == "edge":
            command = build_edge_command(task, safe_message)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("Audio generation timed out after 30 seconds")

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            raise RuntimeError(f"Command failed ({proc.returncode}): {stderr.decode().strip()}")

        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise RuntimeError(f"Audio file not generated or empty: {output_file}")

        volume = task.get("user_volume", 100) / 100.0
        return discord.FFmpegPCMAudio(
            output_file,
            options=f'-af "volume={volume:.2f}"'
        )

    finally:
        print(f"Audio generation took: {time.time() - start_time:.2f}s")

def build_gtts_command(task: dict, safe_message: str) -> str:
    """Build gTTS CLI command with sanitization"""
    tld = task.get("tld", "com")
    return f'gtts-cli "{safe_message}" --tld {tld} --output "{task["debug_mp3"]}"'

def build_edge_command(task: dict, safe_message: str) -> str:
    """Build edge-tts CLI command with pitch/volume controls"""
    return (
        f'edge-tts --pitch {task["edgepitch"]:+}Hz '
        f'--volume {task["edgevolume"]:+}% '
        f'--voice "{task["edge_voice"]}" '
        f'--text "{safe_message}" '
        f'--write-media "{task["debug_mp3"]}"'
    )


async def process_queue():
    global IS_PLAYING

    async with QUEUE_LOCK:
        if tts_queue and not IS_PLAYING:
            IS_PLAYING = True
            task, future = tts_queue.pop(0)
            try:
                source = await future
                # Define cleanup function
                def cleanup(error):
                    global IS_PLAYING
                    IS_PLAYING = False
                    if error:
                        print(f"Playback error for \"{task['content'][:50]}\"...: {error}")
                        if task["retry_count"] < 3:
                            task["retry_count"] += 1
                            new_future = asyncio.create_task(generate_audio(task))
                            tts_queue.insert(0, (task, new_future))
                        else:
                            print(f"Task \"{task['content'][:50]}\"... exceeded retry limit, skipping")
                    else:
                        output_file = task.get("debug_mp3") or task.get("debug_wav")
                        if output_file and os.path.exists(output_file):
                            try:
                                os.remove(output_file)
                            except Exception as e:
                                print(f"Cleanup error: {str(e)}")
                    asyncio.run_coroutine_threadsafe(process_queue(), bot.loop)
                voice_client.play(source, after=cleanup)
                print(f"Now playing: \"{task['content'][:50]}\"...")
            except Exception as e:
                print(f"Error generating audio for \"{task['content'][:50]}\"...': {str(e)}")
                traceback.print_exc()
                if task["retry_count"] < 3:
                    task["retry_count"] += 1
                    new_future = asyncio.create_task(generate_audio(task))
                    tts_queue.insert(0, (task, new_future))
                else:
                    print(f"Task \"{task['content'][:50]}\" exceeded retry limit, skipping")
                IS_PLAYING = False
                await asyncio.sleep(0.1)


def filter_acronyms(content: str) -> str:
    global ACRONYM_CACHE, ACRONYM_LAST_MODIFIED
    
    try:
        file_path = Path("./config/acronyms.json")
        current_mtime = file_path.stat().st_mtime
        
        if current_mtime > ACRONYM_LAST_MODIFIED:
            if not file_path.exists():
                return content
                
            with open(file_path, "r") as f:
                data = json.load(f)
                acronyms = data.get("acronym_replacements", {})
                
            ACRONYM_CACHE = {
                re.compile(rf'\b{re.escape(k)}\b', re.IGNORECASE): v
                for k, v in acronyms.items()
            }
            ACRONYM_LAST_MODIFIED = current_mtime
        
        if ACRONYM_CACHE:
            for pattern, replacement in ACRONYM_CACHE.items():
                content = pattern.sub(replacement, content)
                
        return content
        
    except Exception as e:
        print(f"Error processing acronyms: {str(e)}")
        traceback.print_exc()
        return content

# ----------------------------------
# On Message Event
# ----------------------------------
@bot.event
async def on_message(message):
    if message.channel.id != TEXT_CHANNEL_ID or message.author.bot:
        return

    try:
        userconfig = load_user_config(str(message.author.id))
        if userconfig.get("ignoreme", False):
            return

        processed_content = message.content
        processed_content = replace_mentions(processed_content, message.guild)
        processed_content = filter_acronyms(processed_content)
        processed_content = clean_special_content(processed_content)

        # Handle image and other file attachments
        image_attachments = [
            att for att in message.attachments 
            if att.content_type and att.content_type.startswith('image/')
        ]
        member = message.guild.get_member(int(message.author.id))
        display_name = member.display_name if member else "User"

        if image_attachments:
            if not processed_content.strip():
                processed_content = f"{display_name} sent an image file"
            else:
                processed_content = f"{display_name} sent an image file and said... {processed_content}"
        elif message.attachments:
            if not processed_content.strip():
                processed_content = f"{display_name} sent a file"
            else:
                processed_content = f"{display_name} sent a file and said... {processed_content}"

        # Skip processing if content is empty after handling
        if not processed_content.strip():
            return

        # New check: Skip if no letters and no attachments
        if (not re.search(r'[a-zA-Z]', processed_content, re.IGNORECASE) 
            and not message.attachments):
            return

        task = create_tts_task(processed_content, userconfig)
        future = asyncio.create_task(generate_audio(task))
        
        async with QUEUE_LOCK:
            tts_queue.append((task, future))
        
        await process_queue()

    except Exception as e:
        print(f"Error handling message: {str(e)}")
        traceback.print_exc()

def replace_mentions(content: str, guild) -> str:
    def replace_match(match):
        user_id = match.group(1)
        member = guild.get_member(int(user_id))
        if member:
            # Use server nickname if available, otherwise global name
            return f'@{member.display_name}'
        # Fallback if member not found in guild
        return '@unknown-user'
    
    # Match both <@123> and <@!123> mention formats
    return re.sub(r'<@!?(\d+)>', replace_match, content)

def clean_special_content(content: str) -> str:
    # Replace custom emojis with their name
    content = re.sub(r'<(a?):(\w+):\d+>', r'emoji \2', content)
    # Replace URLs with "link"
    content = re.sub(r'https?://\S+|www\.\S+', 'uh link', content, flags=re.IGNORECASE)
    # Replace newlines with pause indicator
    content = re.sub(r'(\n|\\n)+', '... ', content)
    return content.strip()

def create_tts_task(content: str, config: dict) -> dict:
    unique_id = uuid.uuid4().hex
    service = config.get("service", "edge")
    
    task = {
        "id": unique_id,
        "content": content,
        "user_volume": config.get("volume", 100),
        "service": service,
        "retry_count": 0
    }
    
    if service == "gtts":
        task.update({
            "debug_mp3": f"temp_{unique_id}.mp3",
            "tld": config.get("gtts_tld", "com")
        })
    elif service == "edge":
        task.update({
            "debug_mp3": f"temp_{unique_id}.mp3",
            "edge_voice": config.get("selected_edge_voice"),
            "edgepitch": config.get("edgepitch", 0),
            "edgevolume": config.get("edgevolume", 0)
        })
    
    return task

# ----------------------------------
# Add Acronym Command
# ----------------------------------
@bot.tree.command(name="addacronym", description="Add or update an acronym replacement", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(acronym="The acronym to add/update", translation="The phrase to replace it with")
async def addacronym(interaction: discord.Interaction, acronym: str, translation: str):
    try:
        try:
            with open('./config/acronyms.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"acronym_replacements": {}}
        
        data["acronym_replacements"][acronym.lower()] = translation
        
        with open('./config/acronyms.json', 'w') as f:
            json.dump(data, f, indent=4)
            
        await interaction.response.send_message(
            f"Acronym '{acronym}' added/updated: '{translation}'", 
            ephemeral=True
        )
    except Exception as e:
        await interaction.response.send_message(
            f"❌ Error updating acronyms: {str(e)}", 
            ephemeral=True
        )

# ----------------------------------
# Remove Acronym Command
# ----------------------------------
@bot.tree.command(name="removeacronym", description="Remove an acronym replacement", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(acronym="The acronym to remove")
async def removeacronym(interaction: discord.Interaction, acronym: str):
    try:
        try:
            with open('./config/acronyms.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"acronym_replacements": {}}
        
        acronym_lower = acronym.lower()
        if acronym_lower in data["acronym_replacements"]:
            del data["acronym_replacements"][acronym_lower]
            with open('./config/acronyms.json', 'w') as f:
                json.dump(data, f, indent=4)
            message = f"Acronym '{acronym}' removed"
        else:
            message = f"Acronym '{acronym}' not found"
            
        await interaction.response.send_message(message, ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(
            f"❌ Error removing acronym: {str(e)}", 
            ephemeral=True
        )

DEFAULT_USER_CONFIG = {
    "ignoreme": False,
    "service": "edge",  # Unified service field
    "gtts_tld": "us",
    "selected_edge_voice": None,
    "edgepitch": 0,
    "edgevolume": 0,
    "volume": 100
}

def load_user_config(userid: str) -> dict:
    os.makedirs("./config", exist_ok=True)
    if not os.path.exists("./config/usersettings.json"):
        with open("./config/usersettings.json", "w") as f:
            json.dump({}, f)
    
    try:
        with open('./config/usersettings.json', 'r') as f:
            userconfigs = json.load(f)
    except json.JSONDecodeError:
        userconfigs = {}

    user_config = userconfigs.get(str(userid), {}).copy()
    
    # Migrate old config format
    if "usingedge" in user_config or "usinggtts" in user_config:
        if user_config.get("usingedge"):
            user_config["service"] = "edge"
        else:
            user_config["service"] = "gtts"
        
        # Remove old keys
        user_config.pop("usingedge", None)
        user_config.pop("usinggtts", None)
    if "modelselection" in user_config:
        user_config.pop("modelselection", None)
    if "modelspeaker" in user_config:
        user_config.pop("modelspeaker", None)
    if "selectedmaxspeakers" in user_config:
        user_config.pop("selectedmaxspeakers", None)
    if "randomspeaker" in user_config:
        user_config.pop("randomspeaker", None)
    
    return {
        **DEFAULT_USER_CONFIG,
        **user_config
    }

def save_user_config(userid: str, config: dict):
    try:
        with open('./config/usersettings.json', 'r') as f:
            userconfigs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        userconfigs = {}

    userconfigs[str(userid)] = config
    
    with open('./config/usersettings.json', 'w') as f:
        json.dump(userconfigs, f, indent=4)

# ----------------------------------
# Main Command
# ----------------------------------
if __name__ == '__main__':
    bot.run(TOKEN)