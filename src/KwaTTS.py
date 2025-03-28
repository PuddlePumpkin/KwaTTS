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
import platform
import signal
import emoji
from typing import List
from pathlib import Path
from discord.ext import commands
from discord import app_commands

# ----------------------------------
# GLOBALS
# ----------------------------------
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
voice_client = None
tts_queue = []
QUEUE_LOCK = None
IS_PLAYING = False
ACRONYM_CACHE = None
ACRONYM_LAST_MODIFIED = 0
CONNECTION_STATE = False
LAST_VOICE_CHANNEL = None
RECONNECT_ATTEMPTS = 0
MAX_RECONNECT_ATTEMPTS = 3
VOICE_STATE_LOCK = None

async def graceful_shutdown(signame=None):
    """Handle all shutdown tasks properly"""
    global QUEUE_LOCK
    print(f"\nInitiating graceful shutdown ({signame if signame else 'manual'})...")
    
    # Cleanup queued files
    if not QUEUE_LOCK:  # Safety check
        print("Queue lock not initialized!")
        return
    async with QUEUE_LOCK:
        for task, future in tts_queue:
            output_file = task.get("debug_mp3") or task.get("debug_wav")
            if output_file and os.path.exists(output_file):
                try:
                    os.remove(output_file)
                    print(f"Cleaned queued file: {output_file}")
                except Exception as e:
                    print(f"Error cleaning {output_file}: {str(e)}")
        tts_queue.clear()
    
    # Disconnect from all voice channels
    for guild in bot.guilds:
        voice_client = guild.voice_client
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()
            print(f"Disconnected from voice in {guild.name}")
    
    # Close Discord connection
    if not bot.is_closed():
        await bot.close()
        print("Closed Discord connection")
    
    # Cleanup temporary files
    for file in Path(".").glob("temp_*.mp3"):
        try:
            file.unlink()
            print(f"Cleaned up temp file: {file}")
        except Exception as e:
            print(f"Error cleaning {file}: {str(e)}")
    
    # Exit the program
    sys.exit(0)

def setup_signal_handlers():
    """Register signal handlers with the event loop"""
    loop = asyncio.get_event_loop()
    signals = []
    
    # Platform-specific signal handling
    if platform.system() == 'Windows':
        signals = [signal.SIGINT]
    else:
        signals = [signal.SIGINT, signal.SIGTERM]
    
    for sig in signals:
        try:
            loop.add_signal_handler(
                sig,
                lambda sig=sig: asyncio.create_task(graceful_shutdown(sig.name))
            )
            print(f"Registered handler for {sig.name}")
        except NotImplementedError:
            print(f"Signal {sig.name} not supported on this platform")
        except Exception as e:
            print(f"Error registering handler for {sig.name}: {str(e)}")

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

async def attempt_reconnection():
    global CONNECTION_STATE, RECONNECT_ATTEMPTS, LAST_VOICE_CHANNEL, MAX_RECONNECT_ATTEMPTS
    
    async with VOICE_STATE_LOCK:
        if not CONNECTION_STATE or RECONNECT_ATTEMPTS >= MAX_RECONNECT_ATTEMPTS:
            return

        RECONNECT_ATTEMPTS += 1
        target_channel = LAST_VOICE_CHANNEL

    try:
        print(f"Attempting reconnect ({RECONNECT_ATTEMPTS}/{MAX_RECONNECT_ATTEMPTS})...")
        
        # Ensure channel still exists
        if not target_channel or not target_channel.guild:
            print("❌ Target channel no longer exists")
            raise ConnectionError("Invalid channel")
            
        voice_client = await target_channel.connect()
        
        async with VOICE_STATE_LOCK:
            CONNECTION_STATE = True
            RECONNECT_ATTEMPTS = 0
            
        print(f"✅ Reconnected to {target_channel.name}")
        return True
        
    except discord.ClientException as e:
        print(f"❌ ClientException: {str(e)}")
    except discord.DiscordServerError as e:
        print(f"❌ Discord Server Error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        traceback.print_exc()

    # Retry logic with backoff    
    if RECONNECT_ATTEMPTS < MAX_RECONNECT_ATTEMPTS:
        delay = min(2 ** RECONNECT_ATTEMPTS, 30)  # Exponential backoff
        print(f"Retrying in {delay} seconds...")
        await asyncio.sleep(delay)
        return await attempt_reconnection()
    else:
        async with VOICE_STATE_LOCK:
            CONNECTION_STATE = False
            LAST_VOICE_CHANNEL = None
        print("❌ Max reconnect attempts reached")
        return False

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
    global CONNECTION_STATE, LAST_VOICE_CHANNEL, RECONNECT_ATTEMPTS
    await interaction.response.send_message(f"Requested join...", ephemeral=True)
    try:
        channel = bot.get_channel(VOICE_CHANNEL_ID)
        if channel:
            human_members = [m for m in channel.members if not m.bot]
            if not human_members:
                await interaction.followup.send("Aborted join (No members present).", ephemeral=True)
                return

            voice_client = interaction.guild.voice_client
            if voice_client and voice_client.is_connected():
                await voice_client.move_to(channel)
            else:
                voice_client = await channel.connect()
            
            async with VOICE_STATE_LOCK:
                CONNECTION_STATE = True
                LAST_VOICE_CHANNEL = channel
                RECONNECT_ATTEMPTS = 0
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
    global CONNECTION_STATE, LAST_VOICE_CHANNEL, QUEUE_LOCK
    if not QUEUE_LOCK:  # Safety check
        print("Queue lock not initialized!")
        return
    await interaction.response.send_message(f"Requested leave...", ephemeral=True)
    voice_client = interaction.guild.voice_client
    if voice_client and voice_client.is_connected():
        async with VOICE_STATE_LOCK:
            CONNECTION_STATE = False
            LAST_VOICE_CHANNEL = None
        
        # Cleanup queued files
        async with QUEUE_LOCK:
            for task, future in tts_queue:
                output_file = task.get("debug_mp3") or task.get("debug_wav")
                if output_file and os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                        print(f"Cleaned queued file: {output_file}")
                    except Exception as e:
                        print(f"Error cleaning {output_file}: {str(e)}")
            tts_queue.clear()
        
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
    global VOICE_STATE_LOCK, QUEUE_LOCK
    # Initialize locks after event loop is running
    VOICE_STATE_LOCK = asyncio.Lock()
    QUEUE_LOCK = asyncio.Lock()

    print(f'Bot ready: {bot.user}')
    setup_signal_handlers()
    
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
    """Build absolute path gTTS command with verification"""
    tld = task.get("tld", "com")
    
    # Get path to virtual environment
    venv_path = Path(sys.executable).parent.parent
    
    # Platform-specific paths
    if platform.system() == "Windows":
        cli_path = venv_path / "Scripts" / "gtts-cli.exe"
    else:
        cli_path = venv_path / "bin" / "gtts-cli"
    print(f"Final gtts-cli path: {cli_path}")  # Should show absolute path
    print(f"File exists: {cli_path.exists()}")  # Must be True
    # Verify executable exists
    if not cli_path.exists():
        raise RuntimeError(f"gtts-cli not found at {cli_path}\n"
                          "Install with: pip install gtts")
    
    return (
        f'"{cli_path}" "{safe_message}" '
        f'--tld {tld} --output "{task["debug_mp3"]}"'
    )

def build_edge_command(task: dict, safe_message: str) -> str:
    """Build edge-tts command that works on all platforms"""
    return (
        f'"{sys.executable}" -m edge_tts '
        f'--pitch {task["edgepitch"]:+}Hz '
        f'--volume {task["edgevolume"]:+}% '
        f'--voice "{task["edge_voice"]}" '
        f'--text "{safe_message}" '
        f'--write-media "{task["debug_mp3"]}"'
    )


async def process_queue():
    global IS_PLAYING, QUEUE_LOCK

    if not QUEUE_LOCK:  # Safety check
        print("Queue lock not initialized!")
        return

    async with QUEUE_LOCK:
        voice_client = bot.get_guild(GUILD_ID).voice_client
        if not voice_client or not voice_client.is_connected():
            # Cleanup any remaining queue items
            for task, future in tts_queue:
                output_file = task.get("debug_mp3") or task.get("debug_wav")
                if output_file and os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except Exception as e:
                        print(f"Cleanup error: {str(e)}")
            tts_queue.clear()
            IS_PLAYING = False
            return
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
def contains_code_block(content: str) -> bool:
    """Check if message contains markdown code blocks"""
    return re.search(r'```', content) is not None  # Simplified check

@bot.event
async def on_message(message):
    global QUEUE_LOCK

    if not QUEUE_LOCK:
        return

    if message.channel.id != TEXT_CHANNEL_ID or message.author.bot:
        return

    try:
        userconfig = load_user_config(str(message.author.id))
        if userconfig.get("ignoreme", False):
            return

        # Initial content processing
        processed_content = message.content
        has_code = contains_code_block(processed_content)
        processed_content = re.sub(r'```.*?```', ' ', processed_content, flags=re.DOTALL)
        processed_content = re.sub(r'\s+', ' ', processed_content).strip()

        # Continue processing
        processed_content = replace_mentions(processed_content, message.guild)
        processed_content = filter_acronyms(processed_content)
        processed_content = clean_special_content(processed_content)

        # Attachment analysis
        image_attachments = [att for att in message.attachments 
                           if att.content_type and att.content_type.startswith('image/')]
        file_attachments = [att for att in message.attachments 
                          if att not in image_attachments]
        member = message.guild.get_member(int(message.author.id))
        display_name = member.display_name if member else "User"

        # Attachment counts
        image_count = len(image_attachments)
        file_count = len(file_attachments)
        code_count = 1 if has_code else 0
        total_attachments = image_count + file_count + code_count
        is_long_message = len(processed_content) > 350

        # Build specific description
        specific_attachment = None
        if total_attachments == 1:
            if image_count == 1:
                specific_attachment = "an image"
            elif file_count == 1:
                specific_attachment = "a file"
            elif has_code:
                specific_attachment = "a code block"
        
        # Message construction
        final_content = ""
        if is_long_message:
            base = f"{display_name} sent"
            if total_attachments > 0:
                if specific_attachment and total_attachments == 1:
                    final_content = f"{base} {specific_attachment} and a long message"
                else:
                    final_content = f"{base} multiple attachments and a long message"
            else:
                final_content = f"{base} a long message"
        elif total_attachments > 0:
            base = f"{display_name} sent"
            
            if specific_attachment:
                attachment_desc = specific_attachment
            else:
                attachment_desc = "multiple attachments"
            
            if processed_content:
                final_content = f"{base} {attachment_desc} and said... {processed_content}"
            else:
                final_content = f"{base} {attachment_desc}"
        else:
            final_content = processed_content

        # Validate content
        has_content = any([
            re.search(r'[a-zA-Z0-9]', final_content),
            emoji.emoji_count(final_content) > 0,
            total_attachments > 0
        ])

        if not has_content:
            return

        task = create_tts_task(final_content, userconfig)
        future = asyncio.create_task(generate_audio(task))
        
        async with QUEUE_LOCK:
            tts_queue.append((task, future))
        
        await process_queue()

    except Exception as e:
        print(f"Message processing error: {str(e)}")
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
# Voice state update event (for leaving empty channels)
# ----------------------------------
@bot.event
async def on_voice_state_update(member, before, after):
    global CONNECTION_STATE, VOICE_STATE_LOCK, LAST_VOICE_CHANNEL, QUEUE_LOCK

    if not QUEUE_LOCK:  # Safety check
        print("Queue lock not initialized!")
        return

    if member.id == bot.user.id:
        if before.channel and not after.channel:
            print("⚠️ Bot disconnected from voice channel")
            
            # Check if this was an intentional disconnect
            async with VOICE_STATE_LOCK:
                if not CONNECTION_STATE:
                    print("Clean disconnect, not reconnecting")
                    return
                
                # Maintain connection state for reconnect attempts
                CONNECTION_STATE = True
                
            # Start reconnection in a new task
            asyncio.create_task(handle_reconnection_sequence())

    # Handle human members leaving
    voice_client = member.guild.voice_client
    if voice_client and voice_client.is_connected():
        # Check if there are any non-bot members left in the channel
        human_members = [m for m in voice_client.channel.members if not m.bot]
        if len(human_members) == 0:
            # Disconnect and clean up
            async with VOICE_STATE_LOCK:
                CONNECTION_STATE = False
                LAST_VOICE_CHANNEL = None
            await voice_client.disconnect()
            print("Left voice channel because it became empty.")
            async with QUEUE_LOCK:
                for task, future in tts_queue:
                    output_file = task.get("debug_mp3") or task.get("debug_wav")
                    if output_file and os.path.exists(output_file):
                        try:
                            os.remove(output_file)
                        except Exception as e:
                            print(f"Error cleaning file: {e}")
                tts_queue.clear()

async def handle_reconnection_sequence():
    global QUEUE_LOCK
    if not QUEUE_LOCK:  # Safety check
        print("Queue lock not initialized!")
        return
    print("Starting reconnection sequence...")
    success = await attempt_reconnection()
    
    if not success:
        # Notify in text channel
        channel = bot.get_channel(TEXT_CHANNEL_ID)
        if channel:
            await channel.send("⚠️ Failed to reconnect to voice channel after several attempts. Use `/join` to manually reconnect.")
            
        # Full cleanup
        async with VOICE_STATE_LOCK:
            CONNECTION_STATE = False
            LAST_VOICE_CHANNEL = None
            
        async with QUEUE_LOCK:
            for task, future in tts_queue:
                output_file = task.get("debug_mp3") or task.get("debug_wav")
                if output_file and os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except Exception as e:
                        print(f"Error cleaning file: {e}")
            tts_queue.clear()


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
    try:
        bot.run(TOKEN)
    except KeyboardInterrupt:
        asyncio.run(graceful_shutdown("SIGINT"))