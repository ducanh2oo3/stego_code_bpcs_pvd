
import cv2
import numpy as np
import os
import random
import subprocess
import platform
from scipy.fftpack import dct

def extract_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(frames_dir, exist_ok=True)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:05d}.png"), frame)
        idx += 1
    cap.release()
    return idx

def extract_audio(video_path, audio_path="extracted_audio.aac"):
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", audio_path])

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def compute_complexity_dct(block):
    return np.std(dct2(block))

def estimate_capacity(gray, threshold=10.0, block_size=8):
    h, w = gray.shape
    total = 0
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if compute_complexity_dct(block) > threshold:
                total += 1
    return total

def embed_pvd_parity(gray, bits, threshold=10.0, block_size=8):
    h, w = gray.shape
    idx = 0
    loc_map = []
    mod = gray.copy()
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if idx >= len(bits):
                break
            block = mod[i:i+block_size, j:j+block_size]
            if compute_complexity_dct(block) > threshold:
                p1 = int(block[0,0])
                p2 = int(block[0,1])
                b = int(bits[idx])
                # current parity
                if (abs(p1 - p2) % 2) != b:
                    # flip one pixel by 1
                    if p2 < 255:
                        p2 += 1
                    else:
                        p2 -= 1
                    block[0,1] = p2
                    mod[i:i+block_size, j:j+block_size] = block
                loc_map.append((i, j))
                idx += 1
    return mod, loc_map, idx

def bits_to_text(bits):
    chars = []
    for k in range(0, len(bits) - 7, 8):
        chars.append(chr(int(bits[k:k+8], 2)))
    return ''.join(chars)

def main():
    video = input("Nhập đường dẫn video đầu vào: ")
    frames_dir = "frames_temp"
    audio = "extracted_audio.aac"
    count = extract_frames(video, frames_dir)
    fps = get_video_fps(video)
    extract_audio(video, audio)

    frame_idx = random.randint(0, count - 1)
    print(f"Chọn frame: {frame_idx}")
    img = cv2.imread(f"{frames_dir}/frame_{frame_idx:05d}.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cap = estimate_capacity(gray)
    print(f"🔐 Block nhiễu đủ dùng: {cap} bits (1 bit/block)")
    max_chars = cap // 8
    print(f"🔐 Dung lượng tối đa ~{max_chars} ký tự")
    msg = input(f"Nhập thông điệp (<= {max_chars} ký tự): ")
    bits = ''.join(format(ord(c), '08b') for c in msg)

    if len(bits) > cap:
        print("❌ Thông điệp quá dài!")
        return

    mod, locs, used = embed_pvd_parity(gray, bits)
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:05d}.png", cv2.cvtColor(mod, cv2.COLOR_GRAY2BGR))
    with open("key.txt", "w") as f:
        f.write(f"frame_index: {frame_idx}\n")
        f.write(f"locations: {locs}\n")
        f.write(f"bit_length: {used}\n")

    tmp = "stego_noaudio.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{frames_dir}/frame_%05d.png",
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", tmp
    ])
    out = "output_stego.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp, "-i", audio,
        "-c:v", "copy", "-c:a", "aac", out
    ])
    print(f"✅ Video stego: {out}")
if __name__ == "__main__":
    main()
