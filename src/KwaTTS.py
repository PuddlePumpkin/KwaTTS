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
import psutil
from io import BytesIO
import edge_tts
from gtts import gTTS

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
CURRENT_TASK = None
keepalive_task = None

# Enhanced graceful shutdown to cancel all tasks
async def graceful_shutdown(signame=None):
    """Handle shutdown by cancelling all tasks"""
    print(f"\nShutting down ({signame if signame else 'manual'})...")
    
    # Cancel all ongoing tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    # Wait for tasks to handle cancellation
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Cleanup voice clients
    for guild in bot.guilds:
        if guild.voice_client:
            await guild.voice_client.disconnect()
    
    await bot.close()
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

# Load TTS voices
with open('./src/EdgeVoicesSimplified.json', 'r') as f:
    edge_voices_data = json.load(f)

# Load GTTS voices 
with open('./src/GttsVoices.json', 'r') as f:
    gtts_voices_data = json.load(f)

# ----------------------------------
# Change Voice Edge Command
# ----------------------------------
@bot.tree.command(name="change_voice_edge", description="Set your Edge TTS voice", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(voice="The Edge TTS voice to use")
async def changevoiceedge(interaction: discord.Interaction, voice: str):
    # Check if the selected display name exists in our voice data
    if voice not in edge_voices_data.values():
        await interaction.response.send_message(f"Invalid voice: {voice}", ephemeral=True)
        return
    
    await interaction.response.send_message(f"Edge voice set to {voice}", ephemeral=True)
    userconfig = load_user_config(str(interaction.user.id))
    userconfig["service"] = "edge"
    userconfig["selected_edge_voice"] = voice
    save_user_config(str(interaction.user.id), userconfig)

@changevoiceedge.autocomplete("voice")
async def edge_voice_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    global edge_voices_data
    return [
        app_commands.Choice(name=display_name, value=full_id)
        for display_name, full_id in edge_voices_data.items()
        if current.lower() in display_name.lower()
    ][:25]

# ----------------------------------
# Change Voice Other Command (Admin)
# ----------------------------------
@bot.tree.command(name="change_voice_other", description="[Admin] Set another user's Edge TTS voice", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(
    user="User to modify",
    voice="The Edge TTS voice to use"
)
@app_commands.default_permissions(administrator=True)
async def changevoiceother(interaction: discord.Interaction, user: discord.Member, voice: str):
    # Verify administrator permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("❌ You need administrator privileges to use this command.", ephemeral=True)
        return

    # Validate voice selection
    if voice not in edge_voices_data.values():
        await interaction.response.send_message(f"Invalid voice: {voice}", ephemeral=True)
        return

    # Update target user's config
    userconfig = load_user_config(str(user.id))
    userconfig["service"] = "edge"
    userconfig["selected_edge_voice"] = voice
    save_user_config(str(user.id), userconfig)

    await interaction.response.send_message(
        f"✅ {user.display_name}'s Edge voice has been set to {voice}",
        ephemeral=True
    )

@changevoiceother.autocomplete("voice")
async def edge_voice_other_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    # Reuse the same autocomplete as personal voice selection
    return await edge_voice_autocomplete(interaction, current)
# ----------------------------------
# Change Voice GTTS Command
# ----------------------------------
@bot.tree.command(name="change_voice_gtts", description="Switch to Google TTS with specified accent", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(accent="Accent for voice")
async def changevoicegtts(interaction: discord.Interaction, accent: str):
    # Validate against actual TLDs from JSON
    valid_tlds = [v["tld"] for v in gtts_voices_data]
    if accent not in valid_tlds:
        await interaction.response.send_message(f"Invalid accent: {accent}", ephemeral=True)
        return
    
    userconfig = load_user_config(str(interaction.user.id))
    userconfig["service"] = "gtts"
    userconfig["gtts_tld"] = accent.lower()
    save_user_config(str(interaction.user.id), userconfig)
    await interaction.response.send_message(
        f"Switched to Google TTS with {accent} domain", 
        ephemeral=True
    )

@changevoicegtts.autocomplete("accent")
async def gtts_voice_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    global gtts_voices_data
    return [
        app_commands.Choice(name=voice["accent"], value=voice["tld"])
        for voice in gtts_voices_data
        if current.lower() in voice["accent"].lower() or current.lower() in voice["tld"].lower()
    ][:25]

# ----------------------------------
# Universal Settings Command
# ----------------------------------
@bot.tree.command(name="user_settings", description="Adjust your TTS preferences", guild=discord.Object(id=GUILD_ID))
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
# Universal Settings Other Command (Admin)
# ----------------------------------
@bot.tree.command(name="user_settings_other", description="[Admin] Adjust another user's TTS preferences", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(
    user="User to modify",
    volume="Output volume (0-100)",
    ignoreme="Whether to ignore their messages",
    pitchoffset="Edge TTS pitch offset (-30 to 30)",
    volumeoffset="Edge TTS volume offset (-50 to 50)"
)
@app_commands.default_permissions(administrator=True)
async def user_settings_other(
    interaction: discord.Interaction,
    user: discord.Member,
    volume: app_commands.Range[int, 0, 100] = None,
    ignoreme: bool = None,
    pitchoffset: app_commands.Range[int, -30, 30] = None,
    volumeoffset: app_commands.Range[int, -50, 50] = None
):
    # Verify administrator permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ You need administrator privileges to use this command.",
            ephemeral=True
        )
        return

    # Load target user's config
    userconfig = load_user_config(str(user.id))
    changes = []

    # Update each provided parameter
    if volume is not None:
        userconfig["volume"] = volume
        changes.append(f"🔊 Volume: {volume}%")
    if ignoreme is not None:
        userconfig["ignoreme"] = ignoreme
        changes.append(f"👤 IgnoreMe: {'Enabled' if ignoreme else 'Disabled'}")
    if pitchoffset is not None:
        userconfig["edgepitch"] = pitchoffset
        changes.append(f"🎚️ Edge Pitch Offset: {pitchoffset:+}")
    if volumeoffset is not None:
        userconfig["edgevolume"] = volumeoffset
        changes.append(f"🔈 Edge Volume Offset: {volumeoffset:+}%")

    # Save changes
    save_user_config(str(user.id), userconfig)

    # Prepare response
    if not changes:
        response = "No changes specified."
    else:
        response = f"✅ Updated settings for {user.display_name}:\n" + "\n".join(changes)
    
    await interaction.response.send_message(response, ephemeral=True)

# ----------------------------------
# toggle ignoreme
# ----------------------------------
@bot.tree.command(name="toggle_ignore_me", description="Toggle whether your messages are read", guild=discord.Object(id=GUILD_ID))
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
@bot.tree.command(name="edge_settings", description="Set your Edge TTS pitch and volume offsets", guild=discord.Object(id=GUILD_ID))
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
    global CONNECTION_STATE, LAST_VOICE_CHANNEL, RECONNECT_ATTEMPTS, keepalive_task
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
            # Start keepalive
            if keepalive_task:
                keepalive_task.cancel()
            keepalive_task = asyncio.create_task(voice_keepalive(voice_client))
        else:
            print("❌ Voice channel not found")
    except Exception as e:
        print(f"❌ Voice connection failed: {e}")

# ----------------------------------
# Leave Command
# ----------------------------------
@bot.tree.command(name="leave", description="Requests bot to leave voice", guild=discord.Object(id=GUILD_ID))
async def leave(interaction: discord.Interaction):
    global CONNECTION_STATE, LAST_VOICE_CHANNEL, QUEUE_LOCK, keepalive_task
    if not QUEUE_LOCK:  # Safety check
        print("Queue lock not initialized!")
        return
    await interaction.response.send_message(f"Requested leave...", ephemeral=True)
    voice_client = interaction.guild.voice_client
    if voice_client and voice_client.is_connected():
        # Stop keepalive
        if keepalive_task:
            keepalive_task.cancel()
            keepalive_task = None

        async with VOICE_STATE_LOCK:
            CONNECTION_STATE = False
            LAST_VOICE_CHANNEL = None
        
        # Cleanup queued files
        async with QUEUE_LOCK:
            tts_queue.clear()
        
        await voice_client.disconnect()
        print(f"✅ Left voice channel: {voice_client.channel.name}")
    else:
        print("❌ Bot is not in a voice channel")



# ----------------------------------
# Ready Event
# ----------------------------------
@bot.event
async def on_ready():
    global VOICE_STATE_LOCK, QUEUE_LOCK, process_task
    VOICE_STATE_LOCK = asyncio.Lock()
    QUEUE_LOCK = asyncio.Lock()
    
    # Start the process_queue loop
    process_task = bot.loop.create_task(process_queue())
    
    print(f'Bot ready: {bot.user}')
    setup_signal_handlers()
    try:
        guild = discord.Object(id=GUILD_ID)
        print(f"Commands in tree: {[cmd.name for cmd in bot.tree.get_commands(guild=guild)]}")
        synced = await bot.tree.sync(guild=guild)
        print(f"✅ Synced commands: {[cmd.name for cmd in synced]}")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")


# ----------------------------------
# Stopreading Command (current message only)
# ----------------------------------
@bot.tree.command(name="stop_reading", description="Interrupt the currently playing message", guild=discord.Object(id=GUILD_ID))
async def stopreading(interaction: discord.Interaction):
    voice_client = interaction.guild.voice_client
    if not voice_client or not voice_client.is_connected():
        await interaction.response.send_message("❌ Bot is not connected to voice.", ephemeral=True)
        return

    response = []
    if voice_client.is_playing():
        voice_client.stop()
        response.append("⏹️ Stopped current playback")
    
    await interaction.response.send_message("\n".join(response) if response else "❌ No audio playing", ephemeral=True)

# ----------------------------------
# Clearqueue Command (stop + clear queue)
# ----------------------------------
@bot.tree.command(name="clear_queue", description="Empty the message queue and stop playback", guild=discord.Object(id=GUILD_ID))
async def clearqueue(interaction: discord.Interaction):
    global IS_PLAYING, QUEUE_LOCK
    
    await interaction.response.defer(ephemeral=True)
    voice_client = interaction.guild.voice_client
    
    if not voice_client or not voice_client.is_connected():
        await interaction.followup.send("❌ Bot is not connected to voice.", ephemeral=True)
        return

    async with QUEUE_LOCK:
        # Cancel all pending tasks
        for _, future in tts_queue:
            if not future.done():
                future.cancel()
        tts_queue.clear()
        
        # Stop current playback
        if voice_client.is_playing():
            voice_client.stop()
            
        IS_PLAYING = False

    await interaction.followup.send("✅ Queue cleared and playback stopped", ephemeral=True)


async def generate_audio(task: dict) -> discord.AudioSource:
    """Generate audio with buffering optimization for Edge TTS"""
    proc = None
    try:
        start_time = time.time()
        service = task.get("service", "edge")
        volume = task.get("user_volume", 100) / 100.0

        # Common FFmpeg command
        ffmpeg_command = [
            'ffmpeg', '-nostdin', '-y',
            '-f', 'mp3',
            '-i', 'pipe:0',
            '-af', f'volume={volume}',
            '-f', 's16le',
            '-ar', '48000',
            '-ac', '2',
            'pipe:1'
        ]

        proc = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Check if streams are available
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("FFmpeg streams not initialized")
        print("Starting TTS conversion")

        if service == "gtts":
            # Google TTS (single write)
            tts = gTTS(
                text=task["content"],
                lang='en',
                tld=task.get("tld", "com")
            )
            with BytesIO() as mp3_buffer:
                tts.write_to_fp(mp3_buffer)
                mp3_buffer.seek(0)
                audio_data = mp3_buffer.read()

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=audio_data),
                        timeout=30
                    )
                except asyncio.TimeoutError:
                    await proc.kill()
                    raise RuntimeError("gTTS processing timed out")

        elif service == "edge":
            # Edge TTS (non-streaming buffer)
            communicate = edge_tts.Communicate(
                text=task["content"],
                voice=task["edge_voice"],
                pitch=f"{task.get('edgepitch', 0):+}Hz",
                volume=f"{task.get('edgevolume', 0):+}%"
            )

            try:
                # Get the full audio data
                audio_data = b""
                async for chunk in communicate.stream():
                     if chunk["type"] == "audio":
                        audio_data += chunk["data"]

                # Pass the full audio data to FFmpeg
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=audio_data),
                    timeout=30
                )

            except asyncio.TimeoutError:
                await proc.kill()
                raise RuntimeError("Edge TTS generation timed out")
            except Exception as e:
                # Catch other potential errors during generation
                raise RuntimeError(f"Edge TTS generation failed: {e}")


        print("TTS conversion completed")

        # Error checking
        if proc.returncode != 0:
            err_msg = stderr.decode().strip()
            raise RuntimeError(f"FFmpeg failed ({proc.returncode}): {err_msg}")

        return discord.PCMAudio(BytesIO(stdout))

    except Exception as e:
        print(f"Generation error: {str(e)}")
        if proc:
            try:
                # Ensure FFmpeg process is terminated on error
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5) # Wait for termination
            except (ProcessLookupError, asyncio.TimeoutError):
                 # If terminate fails or times out, try killing
                 try:
                     proc.kill()
                     await asyncio.wait_for(proc.wait(), timeout=5)
                 except (ProcessLookupError, asyncio.TimeoutError):
                     pass # Process is likely already gone

        raise # Re-raise the exception after cleanup
    finally:
        if proc and proc.returncode is None: # Check if process is still running
             try:
                 proc.terminate()
                 await asyncio.wait_for(proc.wait(), timeout=5)
             except (ProcessLookupError, asyncio.TimeoutError):
                 try:
                     proc.kill()
                     await asyncio.wait_for(proc.wait(), timeout=5)
                 except (ProcessLookupError, asyncio.TimeoutError):
                     pass

        print(f"Audio generation took: {time.time() - start_time:.2f}s")


async def process_queue():
    global IS_PLAYING, CURRENT_TASK, QUEUE_LOCK
    while True:
        #DEBUG print("process_queue loop running")
        async with QUEUE_LOCK:
            voice_client = bot.get_guild(GUILD_ID).voice_client
            connected = voice_client and voice_client.is_connected()
            #DEBUG print(f"Queue state: connected={connected}, queue_len={len(tts_queue)}, IS_PLAYING={IS_PLAYING}")
            if not connected or not tts_queue or IS_PLAYING:
                await asyncio.sleep(0.1)
                continue
            IS_PLAYING = True
            task, future = tts_queue.pop(0)
            print(f"Processing task: {task['content'][:50]}...")

        try:
            # Generate audio when processing the task
            future = asyncio.create_task(generate_audio(task))
            print(f"Starting audio generation for task: {task['id']}")
            source = await asyncio.wait_for(future, timeout=30.0)
            print(f"Audio generated for task: {task['id']}")
            CURRENT_TASK = task

            def cleanup(error):
                global IS_PLAYING
                if error:
                    print(f"Playback stopped with error: {error}")
                else:
                    print("Playback finished normally")
                IS_PLAYING = False

            voice_client.play(source, after=cleanup)
            print(f"Now playing: \"{task['content'][:50]}\"...")

        except asyncio.CancelledError:
            # Handle cancellation during shutdown
            raise
        except Exception as e:
            print(f"Error processing task: {str(e)}")
            traceback.print_exc()
            async with QUEUE_LOCK:
                IS_PLAYING = False
                if task["retry_count"] < 3:
                    task["retry_count"] += 1
                    tts_queue.insert(0, task)
        except asyncio.TimeoutError:
            print(f"Audio generation timed out for task: {task['id']}")
            raise
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
    return re.search(r'```.*?```', content, flags=re.DOTALL) is not None

@bot.event
async def on_message(message):

    global QUEUE_LOCK

    print(f"Received message from {message.author} in channel {message.channel.id}: {message.content}")
    if not QUEUE_LOCK:
        print("Queue lock not initialized!")
        return
    if message.channel.id != TEXT_CHANNEL_ID or message.author.bot:
        print("Message skipped: wrong channel or bot author")
        return
    voice_client = message.guild.voice_client
    if not (voice_client and voice_client.is_connected()):
        print("Message skipped: bot not connected to voice")
        return

    if message.channel.id != TEXT_CHANNEL_ID or message.author.bot:
        return
        
    voice_client = message.guild.voice_client
    if not (voice_client and voice_client.is_connected()):
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
        stickers = [sticker.name for sticker in message.stickers]
        sticker_count = len(stickers)
        member = message.guild.get_member(int(message.author.id))
        raw_name = member.display_name if member else "User"
        # Filter out non-alphabetic characters from the name
        filtered_name = ''.join([c for c in raw_name if c.isalpha()])
        display_name = filtered_name if filtered_name else "User"

        # Attachment counts
        image_count = len(image_attachments)
        file_count = len(file_attachments)
        code_count = 1 if has_code else 0
        total_attachments = image_count + file_count + code_count + sticker_count
        is_long_message = len(processed_content) > 400

        # Build specific description
        specific_attachment = None
        if total_attachments == 1:
            if image_count == 1:
                specific_attachment = "an image"
            elif file_count == 1:
                specific_attachment = "a file"
            elif has_code:
                specific_attachment = "a code block"
            elif sticker_count == 1:
                specific_attachment = f"a sticker: {stickers[0]}"
        
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

        # Create audio generation future immediately
        future = asyncio.create_task(generate_audio(task))

        async with QUEUE_LOCK:
            tts_queue.append((task, future))
            print(f"Task added to queue, length: {len(tts_queue)}")

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
    content = re.sub(r'https?://\S+|www\.\S+', 'a link', content, flags=re.IGNORECASE)

    # Demojize Normal Emoji
    content = emoji.demojize(content)
    content = re.sub(r':(\w+):', lambda m: m.group(1).replace('_', ' '), content)

    # Replace newlines with pause indicator
    content = re.sub(r'(\n|\\n)+', '... ', content)
    return content.strip()

def create_tts_task(content: str, config: dict) -> dict:
    """Create a TTS task without file references"""
    return {
        "id": uuid.uuid4().hex,
        "content": content,
        "user_volume": config.get("volume", 100),
        "service": config.get("service", "edge"),
        "edge_voice": config.get("selected_edge_voice"),
        "edgepitch": config.get("edgepitch", 0),
        "edgevolume": config.get("edgevolume", 0),
        "tld": config.get("gtts_tld", "com"),
        "retry_count": 0
    }

# ----------------------------------
# Add Acronym Command
# ----------------------------------
@bot.tree.command(name="add_acronym", description="Create new acronym expansion", guild=discord.Object(id=GUILD_ID))
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
    
    print(f"DEBUG: on_voice_state_update triggered for {member.display_name}. Current CONNECTION_STATE: {CONNECTION_STATE}") # Log entry state
    if not QUEUE_LOCK:  # Safety check
        print("Queue lock not initialized!")
        return

    # Check if the bot itself was disconnected
    if member.id == bot.user.id:
        if before.channel and not after.channel:
            print("⚠️ Bot disconnected from voice channel")
            async with QUEUE_LOCK:
                IS_PLAYING = False
            
            async with VOICE_STATE_LOCK:
                print(f"DEBUG: Bot disconnected. Checking CONNECTION_STATE. Value is: {CONNECTION_STATE}") # Log state before check
                if not CONNECTION_STATE:
                    print("Clean disconnect, not reconnecting")
                    return
                # Maintain connection state for reconnect attempts
                CONNECTION_STATE = True
            # Start reconnection
            asyncio.create_task(handle_reconnection_sequence())

    # Handle human members leaving (only if the bot is connected)
    voice_client = member.guild.voice_client
    if voice_client and voice_client.is_connected() and member.id != bot.user.id:
        await asyncio.sleep(10)  # 10-second cooldown

        # Get current voice client after delay
        current_voice_client = member.guild.voice_client
        if current_voice_client and current_voice_client.is_connected():
            current_channel = current_voice_client.channel
            human_members = [m for m in current_channel.members if not m.bot]
            print(f"DEBUG: Human activity detected ({member.display_name}). Re-checking members in {current_channel.name}. Found {len(human_members)} humans.") # Log check

            if len(human_members) == 0:
                print(f"DEBUG: Entering 'leave because empty' block triggered by activity from {member.display_name}.") # Log entry to this block
                async with VOICE_STATE_LOCK:
                    print(f"DEBUG: Setting CONNECTION_STATE = False (Leave empty)") # Log state change
                    CONNECTION_STATE = False
                    LAST_VOICE_CHANNEL = None
                await current_voice_client.disconnect()
                print("Left voice channel because it became empty.")
                async with QUEUE_LOCK:
                    tts_queue.clear()

async def handle_reconnection_sequence():
    global QUEUE_LOCK
    if not QUEUE_LOCK:  # Safety check
        print("Queue lock not initialized!")
        return
    print("Starting reconnection sequence...")
    success = await attempt_reconnection()
    if success:
        # Removed the call to await process_queue() here.
        # The main process_queue loop (started in on_ready)
        # will automatically resume processing the queue now that
        # the bot is reconnected.
        print("Reconnection successful, main queue process will resume.")
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
            tts_queue.clear()


# ----------------------------------
# Remove Acronym Command
# ----------------------------------
@bot.tree.command(name="remove_acronym", description="Delete an existing acronym", guild=discord.Object(id=GUILD_ID))
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
    "selected_edge_voice": "en-GB-RyanNeural",
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


async def voice_keepalive(voice_client):
    """Keeps voice connection alive with proper silence packets"""
    print("🔊 Starting voice keepalive")
    while voice_client.is_connected():
        if not voice_client.is_playing():
            # Send valid OPUS silence frame
            voice_client.send_audio_packet(b'\xF8\xFF\xFE', encode=False)
        await asyncio.sleep(15)
    print("🔊 Ending voice keepalive")

# ----------------------------------
# Main Command
# ----------------------------------
if __name__ == '__main__':
    try:
        bot.run(TOKEN)
    except KeyboardInterrupt:
        asyncio.run(graceful_shutdown("SIGINT"))