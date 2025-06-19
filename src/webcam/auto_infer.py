#!/usr/bin/env python3
# src/webcam/auto_infer_end.py

import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# ─ argparse 세팅 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Sign2Text 자동 모드 (영상 끝나고 한 번에 예측)"
)
parser.add_argument('video_name', nargs='?', default='일상생활_수어_사회생활_경찰서.mkv', help="videos/ 폴더의 파일명 (예: demo.mp4). 기본값: demo.mp4")
parser.add_argument('--seq',   default='L20', help="윈도우 시퀀스 (기본 L20)")
parser.add_argument('--conf',  type=float, default=0.3,  help="신뢰도 문턱값")
parser.add_argument('--temp',  type=float, default=2.5,  help="온도 스케일링 T")
args = parser.parse_args()

SEQ_NAME    = args.seq
WINDOW_SIZE = int(SEQ_NAME[1:])
CONF_THRESH = args.conf
T           = args.temp

# ─ 모델 및 메타데이터 로드 ───────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODEL_DIR     = os.path.join(BASE_DIR, 'models', SEQ_NAME)
VIDEOS_DIR    = os.path.join(BASE_DIR, 'videos')
model         = load_model(os.path.join(MODEL_DIR, 'sign_language_model_normalized.h5'))
label_classes = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
X_mean        = np.load(os.path.join(MODEL_DIR, 'X_mean.npy'))
X_std         = np.load(os.path.join(MODEL_DIR, 'X_std.npy'))
id2label      = {i: lbl for i, lbl in enumerate(label_classes)}

# ─ 비디오 파일 탐색 ────────────────────────────────────────────────────────────
VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')
name, ext = os.path.splitext(args.video_name)
if ext.lower() in ('.mp4', '.mkv'):
    video_file = os.path.join(VIDEOS_DIR, args.video_name)
else:
    # 확장자 미지정 시 .mp4 → .mkv 순으로 시도
    if os.path.exists(os.path.join(VIDEOS_DIR, name + '.mp4')):
        video_file = os.path.join(VIDEOS_DIR, name + '.mp4')
    elif os.path.exists(os.path.join(VIDEOS_DIR, name + '.mkv')):
        video_file = os.path.join(VIDEOS_DIR, name + '.mkv')
    else:
        print(f"❌ 비디오 파일이 존재하지 않습니다: {args.video_name}")
        exit(1)

# ─ MediaPipe 손 검출 세팅 ──────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_rel(lms, W, H):
    if not lms: return [0]*42
    pts = [(p.x*W, p.y*H) for p in lms]
    bx, by = pts[0]
    rel = []
    for x,y in pts: rel += [x-bx, y-by]
    return rel

def calc_ang(lms):
    if not lms: return [0]*15
    ang=[]
    for i in range(len(lms)-2):
        a,b,c = np.array([lms[i].x, lms[i].y]), np.array([lms[i+1].x, lms[i+1].y]), np.array([lms[i+2].x, lms[i+2].y])
        ba, bc = a-b, c-b
        cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
        ang.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
    return ang[:15]+[0]*(15-len(ang))

# ─ 비디오 열기 ─────────────────────────────────────────────────────────────────
video_file = os.path.join(VIDEOS_DIR, args.video_name)
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("❌ 비디오 열기 실패:", video_file)
    exit(1)

print(f"▶ `{args.video_name}` 처리 시작 (SEQ={SEQ_NAME}, CONF={CONF_THRESH}, T={T})")
sequence = []

# ─ 프레임 순회하며 특징 추출 ────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret: break

    img = cv2.flip(frame, 1)
    H, W = img.shape[:2]
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    left = right = []
    if res.multi_hand_landmarks:
        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            if hd.classification[0].label=='Left':
                left = lm.landmark
            else:
                right = lm.landmark

    feats = extract_rel(left, W, H) + extract_rel(right, W, H) + calc_ang(left) + calc_ang(right)
    if any(abs(f)>1e-6 for f in feats):
        sequence.append(feats)

cap.release()

# ─ 전체 시퀀스 한 번에 예측 ─────────────────────────────────────────────────────
if len(sequence) < WINDOW_SIZE:
    print("❗ 읽어들인 프레임 수 부족:", len(sequence), "<", WINDOW_SIZE)
    exit(0)

seq_arr   = np.array(sequence, dtype=np.float32)
n_windows = seq_arr.shape[0] - WINDOW_SIZE + 1
windows   = np.stack([seq_arr[i:i+WINDOW_SIZE] for i in range(n_windows)],axis=0)
normed    = (windows - X_mean) / X_std

probs     = model.predict(normed, verbose=0)
logits    = np.log(np.clip(probs, 1e-12,1.0))
scaled    = np.exp(logits / T)
probs_T   = scaled / np.sum(scaled,axis=1,keepdims=True)

# 각 윈도우마다 Top-1 confidence
scores    = probs_T.max(axis=1)
best_idx  = scores.argmax()
best_p    = probs_T[best_idx]
pred      = best_p.argmax()
conf      = best_p[pred]

print("\n=== 최종 예측 결과 ===")
print(f"Top-1: {id2label[pred]} (conf={conf:.3f})")
print("Top-3:")
for idx in best_p.argsort()[-3:][::-1]:
    print(f"  {id2label[idx]}: {best_p[idx]:.3f}")

if conf < CONF_THRESH:
    print("⚠️ 신뢰도 낮음; threshold:", CONF_THRESH)
else:
    print("✅ 예측 완료")

