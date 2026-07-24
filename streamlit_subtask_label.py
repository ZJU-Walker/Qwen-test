import os

import streamlit as st
import streamlit.components.v1 as components # <-- BROUGHT BACK FOR JAVASCRIPT
import json
import re
import av  # PyAV decodes the AV1-encoded mp4s LeRobot writes (libdav1d); no h264 conversion needed

# --- Configuration ---
CHUNK_DIR = "/iris/projects/humanoid/trossen_data/0716_green_block_mem_extra/videos/chunk-000"
VIDEO_DIR = os.path.join(CHUNK_DIR, "observation.images.cam_high")
LABEL_FILE = os.path.join(CHUNK_DIR, "subtask_labels.json")

TASKS = ["waiting", "pick up green block"]

def save_labels():
    """Atomic save: write tmp then os.replace, so a crash/kill can never truncate
    the label file (that's how a morning of labels was lost on 2026-07-17).
    Keeps the previous good version in subtask_labels.json.bak."""
    tmp = LABEL_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(st.session_state.all_labels, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(LABEL_FILE) and os.path.getsize(LABEL_FILE) > 0:
        os.replace(LABEL_FILE, LABEL_FILE + ".bak")
    os.replace(tmp, LABEL_FILE)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# ==========================================
# Button callbacks (run BEFORE the script body on the click's rerun, so state is
# final by render time -- no st.rerun() needed. The old pattern of mutating state
# at the bottom of the script + st.rerun() double-ran the app on every 'r' press,
# which the frontend sometimes rendered as a stuck RUNNING/loading state.)
# ==========================================
def _final_frame_idx():
    return len(st.session_state.video_frames) - 1

def _required_start():
    segs = st.session_state.segments
    return 0 if len(segs) == 0 else segs[-1]['end'] + 1

def _session_ready():
    # A click from a stale browser page can land on a brand-new server session whose
    # state hasn't been built yet (callbacks run before the script body).
    return 'video_frames' in st.session_state and 'segments' in st.session_state

def cb_next_frame():
    if not _session_ready(): return
    st.session_state.frame_idx = min(st.session_state.frame_idx + 1, _final_frame_idx())

def cb_prev_frame():
    if not _session_ready(): return
    st.session_state.frame_idx = max(st.session_state.frame_idx - 1, 0)

def cb_set_task(i):
    st.session_state.active_task_idx = i

def cb_mark_end():
    if not _session_ready(): return
    final = _final_frame_idx()
    rs = _required_start()
    if rs > final:
        st.session_state.flash = ("error", "Video is already fully labeled!")
    elif st.session_state.frame_idx >= rs:
        st.session_state.segments.append({
            "task": TASKS[st.session_state.active_task_idx],
            "start": rs,
            "end": st.session_state.frame_idx
        })
        st.session_state.all_labels[st.session_state.current_video] = st.session_state.segments
        save_labels()
        if st.session_state.frame_idx == final:
            st.toast("✅ Video fully labeled and auto-saved!")
        else:
            st.toast("✅ Segment saved!")
        st.session_state.frame_idx = min(st.session_state.frame_idx + 1, final)
    else:
        st.session_state.flash = ("error", f"Cannot mark end at {st.session_state.frame_idx}. It must be >= {rs}")

def cb_undo():
    if not _session_ready() or not st.session_state.segments: return
    popped = st.session_state.segments.pop()
    st.session_state.all_labels[st.session_state.current_video] = st.session_state.segments
    st.session_state.frame_idx = max(popped['end'], 0)

st.set_page_config(layout="wide", page_title="MP4 Subtask Labeler")

if not os.path.exists(VIDEO_DIR):
    st.error(f"Video directory not found: {VIDEO_DIR}")
    st.stop()

# ==========================================
# 1. State Management Initialization
# ==========================================
if 'active_task_idx' not in st.session_state:
    st.session_state.active_task_idx = 0
if 'frame_idx' not in st.session_state:
    st.session_state.frame_idx = 0
if 'all_labels' not in st.session_state:
    st.session_state.all_labels = {}
    if os.path.exists(LABEL_FILE):
        try:
            with open(LABEL_FILE, 'r') as f:
                st.session_state.all_labels = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt/empty file: quarantine it and keep going instead of crash-looping.
            quarantine = LABEL_FILE + ".corrupt"
            os.replace(LABEL_FILE, quarantine)
            bak = LABEL_FILE + ".bak"
            if os.path.exists(bak):
                try:
                    with open(bak, 'r') as f:
                        st.session_state.all_labels = json.load(f)
                    st.warning(f"Label file was corrupt (moved to {quarantine}); restored from .bak")
                except (json.JSONDecodeError, OSError):
                    st.warning(f"Label file AND .bak were corrupt; starting empty (bad file: {quarantine})")
            else:
                st.warning(f"Label file was corrupt (moved to {quarantine}); starting empty")
    else:
        # File missing entirely (e.g. a crash after quarantine): restore from .bak if we have one.
        bak = LABEL_FILE + ".bak"
        if os.path.exists(bak):
            try:
                with open(bak, 'r') as f:
                    st.session_state.all_labels = json.load(f)
                save_labels()
                st.warning("Label file was missing; restored from .bak")
            except (json.JSONDecodeError, OSError):
                st.warning("Label file missing and .bak unreadable; starting empty")

st.title("🎬 MP4 High-Speed Subtask Labeler")

# Find all MP4s
videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')], key=natural_sort_key)
if not videos:
    st.error(f"No .mp4 files found in {VIDEO_DIR}")
    st.stop()

selected_video = st.sidebar.selectbox("Select Episode", videos)
video_path = os.path.join(VIDEO_DIR, selected_video)

# ==========================================
# 2. Decode Video to RAM (Fixes Rendering & H264 issues)
# ==========================================
if 'current_video' not in st.session_state or st.session_state.current_video != selected_video:
    with st.spinner(f"Loading {selected_video} into memory for instantaneous scrubbing..."):
        frames = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))

        if not frames:
            st.error("Failed to decode any frames from the video. File may be corrupted.")
            st.stop()
            
        # Store frames directly in session state
        st.session_state.video_frames = frames
        st.session_state.current_video = selected_video
        
        # Load segments for this specific video
        st.session_state.segments = st.session_state.all_labels.get(selected_video, [])
        req_start = 0 if len(st.session_state.segments) == 0 else st.session_state.segments[-1]['end'] + 1
        st.session_state.frame_idx = min(req_start, len(frames) - 1)

final_frame_idx = _final_frame_idx()
required_start = _required_start()

# flash message set by a callback during this rerun (e.g. invalid mark-end)
if (msg := st.session_state.pop("flash", None)) is not None:
    kind, text = msg
    getattr(st, kind)(text)

# ==========================================
# 3. Main UI
# ==========================================
st.sidebar.markdown("""
### ⌨️ Hotkeys
* **Arrow Keys / Scroll**: Next/Prev Frame
* **Keys 1, 2**: Change Subtask
* **Key 'R'**: Mark End & Save Segment
""")
st.sidebar.divider()

def sync_slider():
    st.session_state.frame_idx = st.session_state.slider_val

st.session_state.frame_idx = min(st.session_state.frame_idx, final_frame_idx)

frame_idx = st.slider(
    "Scrub through frames", 
    min_value=0,
    max_value=final_frame_idx, 
    value=st.session_state.frame_idx,
    key="slider_val",
    on_change=sync_slider
)

col1, col2 = st.columns([2, 1])

with col1:
    # Load directly from the RAM cache array - Instantly renders
    frame = st.session_state.video_frames[frame_idx]
    st.image(frame, use_column_width=True, caption=f"Frame: {frame_idx} / {final_frame_idx}")

with col2:
    if required_start > final_frame_idx:
        st.success("🎉 All frames in this video have been labeled!") 
    else:
        selected_task_name = st.radio(
            "Current Subtask (Press 1, 2):", 
            TASKS, 
            index=st.session_state.active_task_idx
        )
        st.session_state.active_task_idx = TASKS.index(selected_task_name)
        
        st.info(f"📍 **Auto-Start:** Frame `{required_start}`")
        st.write(f"🛑 **End (Press 'r'):** Frame `{frame_idx}`")

    st.divider()
    st.header("📋 Recorded Chain")
    for i, seg in enumerate(st.session_state.segments):
        st.write(f"**{i+1}. {seg['task']}** (`{seg['start']}` ➔ `{seg['end']}`)")
        
    if len(st.session_state.segments) > 0:
        st.button("↩️ Undo Last Segment", on_click=cb_undo)

    st.divider()
    is_complete = len(st.session_state.segments) > 0 and st.session_state.segments[-1]['end'] == final_frame_idx
    
    if st.button("💾 SAVE PROGRESS", width="stretch", type="primary"):
        st.session_state.all_labels[selected_video] = st.session_state.segments
        save_labels()
        if is_complete:
            st.success("Saved as Complete!")
        else:
            st.warning("Saved partial progress.")


# ==========================================
# 4. The Keyboard Engine 
# ==========================================
st.sidebar.divider()
st.sidebar.caption("⚙️ Active Hotkey Engine")
with st.sidebar.container():
    st.button("ActionNextFrame", on_click=cb_next_frame)
    st.button("ActionPrevFrame", on_click=cb_prev_frame)

    for i in range(min(len(TASKS), 4)):
        st.button(f"ActionTask{i+1}", on_click=cb_set_task, args=(i,))

    st.button("ActionMarkEnd", on_click=cb_mark_end)

# ==========================================
# 5. JavaScript Injection (Now actually runs)
# ==========================================
components.html(
    """
    <script>
    const doc = window.parent.document;
    
    // Prevent adding multiple identical listeners when Streamlit reruns
    if (!doc.getElementById("labeler_hotkeys_injected")) {
        let marker = doc.createElement("div");
        marker.id = "labeler_hotkeys_injected";
        marker.style.display = "none";
        doc.body.appendChild(marker);

        function clickBtn(text) {
            let btns = Array.from(doc.querySelectorAll('button'));
            // Trim whitespace to ensure exact matches
            let btn = btns.find(b => b.textContent && b.textContent.trim() === text);
            if (btn) btn.click();
        }

        let lastMarkEnd = 0;
        doc.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'TEXTAREA') return;
            if (e.target.tagName === 'INPUT' && e.target.type === 'text') {
                // The episode selectbox is a text input (combobox). Focus lingers in it
                // after picking an episode, which used to swallow ALL hotkeys until the
                // user clicked elsewhere. If its dropdown is open, let it keep native
                // keys; otherwise blur it and handle the hotkey normally.
                if (!e.target.closest('[data-baseweb="select"]')) return;
                if (e.target.getAttribute('aria-expanded') === 'true') return;
                e.target.blur();
            }

            if (e.key === '1') { e.preventDefault(); clickBtn('ActionTask1'); }
            if (e.key === '2') { e.preventDefault(); clickBtn('ActionTask2'); }
            if (e.key === '3') { e.preventDefault(); clickBtn('ActionTask3'); }
            if (e.key === '4') { e.preventDefault(); clickBtn('ActionTask4'); }

            if (e.key.toLowerCase() === 'r') {
                e.preventDefault();
                // Debounce: a double-tap of 'r' used to fire mid-rerun, error out the app,
                // and (worst case) truncate the label file. One mark per 600 ms.
                let now = Date.now();
                if (now - lastMarkEnd < 600) return;
                lastMarkEnd = now;
                clickBtn('ActionMarkEnd');
            }
            
            if (e.key === 'ArrowRight') { e.preventDefault(); clickBtn('ActionNextFrame'); }
            if (e.key === 'ArrowLeft') { e.preventDefault(); clickBtn('ActionPrevFrame'); }
        });

        let lastScroll = 0;
        doc.addEventListener('wheel', function(e) {
            if (e.target.closest('[data-testid="stSidebar"]')) return;
            
            let now = Date.now();
            if (now - lastScroll < 150) return; 
            lastScroll = now;

            if (e.deltaY > 0) clickBtn('ActionNextFrame');
            else if (e.deltaY < 0) clickBtn('ActionPrevFrame');
        });
    }
    </script>
    """,
    height=0, width=0
)