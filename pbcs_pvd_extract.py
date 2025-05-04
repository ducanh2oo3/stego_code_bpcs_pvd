
import cv2
import numpy as np
import ast, re

def read_key(file="key.txt"):
    lines = open(file).read().splitlines()
    fi = int(re.search(r"\d+", lines[0]).group())
    locs = ast.literal_eval(lines[1].split(": ")[1])
    bl = int(re.search(r"\d+", lines[2]).group())
    return fi, locs, bl

def extract_pvd_parity(gray, locs, bit_len, block_size=8):
    bits = ""
    for i,j in locs:
        if len(bits) >= bit_len:
            break
        block = gray[i:i+block_size, j:j+block_size]
        p1 = int(block[0,0])
        p2 = int(block[0,1])
        bits += str(abs(p1 - p2) % 2)
    return bits

def bits_to_text(bits):
    chars = []
    for k in range(0, len(bits) - 7, 8):
        chars.append(chr(int(bits[k:k+8],2)))
    return ''.join(chars)

def main():
    fi, locs, bl = read_key()
    img = cv2.imread(f"frames_temp/frame_{fi:05d}.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bits = extract_pvd_parity(gray, locs, bl)
    text = bits_to_text(bits)
    print("Thông điệp trích xuất được:")
    print(text)
if __name__=="__main__":
    main()
