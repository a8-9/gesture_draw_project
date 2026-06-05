import cv2
import numpy as np
import pickle
import sys
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import gzip
import joblib

from utils import get_hand_landmarks, extract_features, GESTURE_NAMES


class Gesture(Enum):
    PALM = 0
    FIST = 1
    POINT = 2
    OK = 3
    SCISSORS = 4


class DrawMode(Enum):
    BRUSH = auto()
    ERASER = auto()


@dataclass
class AppConfig:
    canvas_width: int = 1280
    canvas_height: int = 720
    
    confidence_threshold: float = 0.55
    paint_smooth_window: int = 5
    mode_switch_duration: float = 0.2  # 进一步降低
    mode_switch_confidence: float = 0.6  # 进一步降低
    
    action_stable_duration: float = 0.5
    action_cooldown: float = 0.8
    
    ema_alpha: float = 0.3
    fist_hold_frames: int = 10
    clear_cooldown: float = 1.0
    
    brush_sizes: list = field(default_factory=lambda: [2, 4, 6, 8, 12])
    eraser_sizes: list = field(default_factory=lambda: [10, 20, 30, 50])
    colors: list = field(default_factory=lambda: [
        (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (0, 165, 255), (128, 0, 128),
        (255, 255, 0), (255, 192, 203)
    ])


class PaintGestureFilter:
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.55):
        self.window = deque(maxlen=window_size)
        self.confidence_threshold = confidence_threshold
        self.last_valid_gesture: Optional[int] = None
    
    def add(self, gesture: int, probs: Optional[np.ndarray] = None) -> Tuple[int, float]:
        self.window.append(gesture)
        
        counts = {}
        for g in self.window:
            counts[g] = counts.get(g, 0) + 1
        
        total = len(self.window)
        dominant = max(counts, key=counts.get)
        confidence = counts[dominant] / total
        
        if probs is not None:
            model_conf = probs[dominant]
            confidence = 0.5 * confidence + 0.5 * model_conf
        
        if confidence < self.confidence_threshold:
            if self.last_valid_gesture is not None:
                return self.last_valid_gesture, confidence
            return dominant, confidence
        
        self.last_valid_gesture = dominant
        return dominant, confidence
    
    def clear(self):
        self.window.clear()
        self.last_valid_gesture = None


class ModeSwitchDetector:
    """
    修复版：去掉绘画锁定，手势之间即时切换
    """
    def __init__(self, switch_duration: float = 0.2, confidence_threshold: float = 0.6):
        self.switch_duration = switch_duration
        self.confidence_threshold = confidence_threshold
        
        self.target_mode: Optional[DrawMode] = None
        self.start_time: float = 0.0
        self.current_mode = DrawMode.BRUSH
    
    def update(self, paint_gesture: int, confidence: float, now: float) -> DrawMode:
        # 确定目标模式
        if paint_gesture == Gesture.SCISSORS.value:
            target = DrawMode.ERASER
        elif paint_gesture == Gesture.POINT.value:
            target = DrawMode.BRUSH
        else:
            self.target_mode = None
            return self.current_mode
        
        # 已经在目标模式
        if target == self.current_mode:
            self.target_mode = None
            return self.current_mode
        
        # 开始切换计时
        if self.target_mode != target:
            self.target_mode = target
            self.start_time = now
        
        elapsed = now - self.start_time
        if elapsed >= self.switch_duration and confidence >= self.confidence_threshold:
            self.current_mode = target
            self.target_mode = None
            print(f"🔄 模式切换: {'橡皮' if target == DrawMode.ERASER else '画笔'}")
            return target
        
        return self.current_mode
    
    def get_progress(self) -> Tuple[float, Optional[DrawMode]]:
        if self.target_mode is None:
            return 0.0, None
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.switch_duration), self.target_mode
    
    def reset(self):
        self.target_mode = None
        self.current_mode = DrawMode.BRUSH


class ActionTrigger:
    ACTION_GESTURES = {Gesture.PALM.value, Gesture.OK.value}
    
    def __init__(self, stable_duration: float = 0.5, cooldown: float = 0.8):
        self.stable_duration = stable_duration
        self.cooldown = cooldown
        
        self.current_gesture: Optional[int] = None
        self.start_time: float = 0.0
        self.triggered: bool = False
        self.last_triggered: Optional[int] = None
        self.last_trigger_time: float = 0.0
        self.stability_window = deque(maxlen=15)
    
    def update(self, gesture: int, now: float) -> Optional[int]:
        self.stability_window.append(gesture)
        
        if len(self.stability_window) < 10:
            return None
        
        counts = {}
        for g in self.stability_window:
            counts[g] = counts.get(g, 0) + 1
        dominant = max(counts, key=counts.get)
        confidence = counts[dominant] / len(self.stability_window)
        
        if confidence < 0.75:
            return None
        
        if dominant not in self.ACTION_GESTURES:
            return None
        
        if dominant != self.current_gesture:
            self.current_gesture = dominant
            self.start_time = now
            self.triggered = False
            return None
        
        duration = now - self.start_time
        if duration < self.stable_duration:
            return None
        
        if not self.triggered:
            if dominant != self.last_triggered or (now - self.last_trigger_time) > self.cooldown:
                self.triggered = True
                self.last_triggered = dominant
                self.last_trigger_time = now
                return dominant
        
        return None
    
    def reset(self):
        self.current_gesture = None
        self.start_time = 0.0
        self.triggered = False
        self.stability_window.clear()


class DrawingCanvas:
    def __init__(self, width: int, height: int, config: AppConfig):
        self.w, self.h = width, height
        self.config = config
        self.draw_layer = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        self.color_idx = 0
        self.mode = DrawMode.BRUSH
        self.paused = False
        self.brush_idx = 1
        self.eraser_idx = 0
        
        self._undo_stack = deque(maxlen=50)
        self._redo_stack = deque(maxlen=50)
        self.save_state()
    
    @property
    def curr_color(self):
        return self.config.colors[self.color_idx % len(self.config.colors)]
    
    @property
    def curr_size(self):
        if self.mode == DrawMode.ERASER:
            return self.config.eraser_sizes[self.eraser_idx % len(self.config.eraser_sizes)]
        return self.config.brush_sizes[self.brush_idx % len(self.config.brush_sizes)]
    
    def save_state(self):
        self._undo_stack.append(self.draw_layer.copy())
        self._redo_stack.clear()
    
    def undo(self) -> bool:
        if len(self._undo_stack) <= 1:
            return False
        self._redo_stack.append(self._undo_stack.pop())
        self.draw_layer = self._undo_stack[-1].copy()
        return True
    
    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        self.draw_layer = state.copy()
        return True
    
    def clear(self):
        self.draw_layer[:] = 255
        self.save_state()
    
    def draw_stroke(self, pts: list):
        if self.paused or len(pts) < 2:
            return
        color = (255, 255, 255) if self.mode == DrawMode.ERASER else self.curr_color
        size = self.curr_size
        for i in range(1, len(pts)):
            cv2.line(self.draw_layer, pts[i-1], pts[i], color, size, cv2.LINE_AA)
    
    def render(self, cursor_pos: Optional[Tuple[int, int]] = None):
        result = self.draw_layer.copy()
        if cursor_pos and not self.paused:
            color = (0, 0, 0) if self.mode == DrawMode.BRUSH else (200, 200, 200)
            cv2.circle(result, cursor_pos, self.curr_size, color, 2)
        self._draw_ui(result)
        return result
    
    def _draw_ui(self, img: np.ndarray):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
        
        mode_text = f"{'ERASER' if self.mode == DrawMode.ERASER else 'BRUSH'} | Size:{self.curr_size}"
        if self.paused:
            mode_text += " [PAUSED]"
        
        cv2.putText(img, mode_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
        cv2.circle(img, (w - 40, 28), 18, self.curr_color, -1)
        cv2.circle(img, (w - 40, 28), 18, (100, 100, 100), 2)


class EMASmoother:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, x: float, y: float) -> Tuple[int, int]:
        pt = np.array([x, y], dtype=np.float32)
        if self.value is None:
            self.value = pt
        else:
            self.value = self.alpha * pt + (1 - self.alpha) * self.value
        return int(self.value[0]), int(self.value[1])
    
    def reset(self):
        self.value = None


class GestureDrawingApp:
    def __init__(self):
        self.config = AppConfig()
        self.canvas = DrawingCanvas(self.config.canvas_width, self.config.canvas_height, self.config)
        
        self.paint_filter = PaintGestureFilter(
            window_size=self.config.paint_smooth_window,
            confidence_threshold=self.config.confidence_threshold
        )
        self.mode_switcher = ModeSwitchDetector(
            switch_duration=self.config.mode_switch_duration,
            confidence_threshold=self.config.mode_switch_confidence
        )
        self.action_trigger = ActionTrigger(
            stable_duration=self.config.action_stable_duration,
            cooldown=self.config.action_cooldown
        )
        
        self.ema = EMASmoother(self.config.ema_alpha)
        
        self.stroke_buffer = []
        self.fist_counter = 0
        self.last_clear_time = 0
        self.is_drawing = False
        
        self.cam_w, self.cam_h = 640, 480
        self.scale_x = self.config.canvas_width / self.cam_w
        self.scale_y = self.config.canvas_height / self.cam_h
        
        self.model = self._load_model()
        self.cap = None
        
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
    
    def _load_model(self):
        try:
            base = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
            model_path = os.path.join(base, "../models/gesture_model.pkl.gz")
            with gzip.open(model_path, "rb") as f:
                model = joblib.load(f)
            print(f"✅ 模型加载成功")
            print(f"   期望特征维度: {model.n_features_in_}")
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def _map_coords(self, landmarks):
        x = landmarks[8].x * self.cam_w * self.scale_x
        y = landmarks[8].y * self.cam_h * self.scale_y
        return self.ema.update(x, y)
    
    def _handle_action(self, gesture: int):
        gesture_name = GESTURE_NAMES[gesture]
        print(f"🎯 触发: {gesture_name}")
        
        if gesture == Gesture.PALM.value:
            self.canvas.paused = not self.canvas.paused
        elif gesture == Gesture.OK.value:
            self.canvas.color_idx = (self.canvas.color_idx + 1) % len(self.config.colors)
    
    def _process_frame(self, frame: np.ndarray):
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        
        frame = cv2.flip(frame, 1)
        landmarks, detected = get_hand_landmarks(frame, int(now * 1000))
        
        if not detected:
            self.paint_filter.clear()
            self.mode_switcher.reset()
            self.action_trigger.reset()
            self.ema.reset()
            if self.is_drawing:
                self._end_stroke()
            
            cv2.putText(frame, "No hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, None
        
        # Layer 1: 原始识别 + 概率
        feat = extract_features(landmarks, enhanced=True).reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(feat)[0]
            raw_pred = int(np.argmax(probs))
        else:
            raw_pred = int(self.model.predict(feat)[0])
            probs = None
        
        # Layer 2: 置信度过滤
        paint_gesture, confidence = self.paint_filter.add(raw_pred, probs)
        
        # Layer 2.5: 模式切换（修复：不再传入 is_drawing）
        new_mode = self.mode_switcher.update(paint_gesture, confidence, now)
        self.canvas.mode = new_mode
        
        # Layer 3: 动作触发
        action = self.action_trigger.update(paint_gesture, now)
        if action is not None:
            self._handle_action(action)
        
        cx, cy = self._map_coords(landmarks)
        
        # 清空画布
        if raw_pred == Gesture.FIST.value:
            self.fist_counter += 1
            if self.fist_counter >= self.config.fist_hold_frames and \
               (now - self.last_clear_time > self.config.clear_cooldown):
                self.canvas.clear()
                self.last_clear_time = now
                self.fist_counter = 0
                self._end_stroke()
                print("🧹 画布已清空")
        else:
            self.fist_counter = max(0, self.fist_counter - 1)
        
        # 绘画逻辑
        can_draw = paint_gesture in (Gesture.POINT.value, Gesture.SCISSORS.value) and not self.canvas.paused
        
        if can_draw:
            self.stroke_buffer.append((cx, cy))
            if len(self.stroke_buffer) >= 2:
                self.canvas.draw_stroke(self.stroke_buffer[-3:])
            self.is_drawing = True
        else:
            if self.is_drawing:
                self._end_stroke()
        
        # 状态显示
        fps = np.mean(self.fps_history) if self.fps_history else 0
        
        status_lines = [
            f"Mode: {'ERASER' if self.canvas.mode == DrawMode.ERASER else 'BRUSH'} | Size:{self.canvas.curr_size}",
            f"Gesture: {GESTURE_NAMES[paint_gesture]} | Conf:{confidence:.2f}",
            f"{'Drawing' if can_draw else 'Idle'} | FPS:{fps:.1f}",
        ]
        
        switch_progress, switch_target = self.mode_switcher.get_progress()
        if switch_progress > 0 and switch_target is not None:
            target_name = "ERASER" if switch_target == DrawMode.ERASER else "BRUSH"
            bar = "█" * int(switch_progress * 8) + "░" * (8 - int(switch_progress * 8))
            status_lines.append(f"Switching: {target_name} [{bar}]")
        
        if self.action_trigger.current_gesture is not None:
            elapsed = now - self.action_trigger.start_time
            progress = min(1.0, elapsed / self.config.action_stable_duration)
            bar = "█" * int(progress * 8) + "░" * (8 - int(progress * 8))
            status_lines.append(f"Action: {GESTURE_NAMES[self.action_trigger.current_gesture]} [{bar}]")
        
        y_offset = 30
        for line in status_lines:
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            y_offset += 22
            
        from PIL import Image, ImageDraw, ImageFont
        
        guide_lines = [
            "[手势指南]",
            "单指：画笔",     
            "剪刀手：橡皮",    
            "握拳(保持2-3秒)：清空画板",   
            "手掌(保持2-3秒)：暂停/恢复",  
            "OK手势：切换颜色"      
        ]
        
        frame_h, frame_w = frame.shape[:2]
        guide_x = frame_w - 240 
        guide_y = 15
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (guide_x - 10, 5), (frame_w - 5, guide_y + len(guide_lines) * 24), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame) # 60% 透明度
        
        # 2. 将 OpenCV 矩阵 (BGR) 转换为 PIL 图片 (RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 3. 自动匹配系统自带的黑体字体（兼容 Windows 和 Mac/Linux）
        font_path = "simhei.ttf" if os.name == 'nt' else "/System/Library/Fonts/STHeiti Light.ttc"
        try:
            # 字体大小设为 16 像素，清晰美观
            font = ImageFont.truetype(font_path, 16)
        except IOError:
            # 如果万一系统字体缺失，降级使用默认流
            font = ImageFont.load_default()
        
        # 4. 逐行写中文
        for i, line in enumerate(guide_lines):
            # 第一行标题用耀眼的黄色，其他用纯白色
            text_color = (255, 255, 0) if i == 0 else (255, 255, 255)
            draw.text((guide_x, guide_y), line, font=font, fill=text_color)
            guide_y += 24
            
        # 5. 把写好中文的 PIL 图片转回 OpenCV 矩阵
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return frame, (cx, cy)
    
    def _end_stroke(self):
        if len(self.stroke_buffer) > 1:
            self.canvas.draw_stroke(self.stroke_buffer)
        self.canvas.save_state()
        self.stroke_buffer.clear()
        self.is_drawing = False
    
    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        
        print("=" * 60)
        print("手势绘图系统（修复版）")
        print("=" * 60)
        print("=" * 60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame, cursor = self._process_frame(frame)
            cv2.imshow("Camera", frame)
            cv2.imshow("Canvas", self.canvas.render(cursor))
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                self.canvas.clear()
            elif key == ord('z'):
                self.canvas.undo()
            elif key == ord('y'):
                self.canvas.redo()
            elif key == ord('s'):
                filename = f"drawing_{int(time.time())}.png"
                cv2.imwrite(filename, self.canvas.render())
                print(f"💾 已保存: {filename}")
            elif key == ord('+'):
                if self.canvas.mode == DrawMode.BRUSH:
                    self.canvas.brush_idx = min(len(self.config.brush_sizes) - 1, self.canvas.brush_idx + 1)
                else:
                    self.canvas.eraser_idx = min(len(self.config.eraser_sizes) - 1, self.canvas.eraser_idx + 1)
            elif key == ord('-'):
                if self.canvas.mode == DrawMode.BRUSH:
                    self.canvas.brush_idx = max(0, self.canvas.brush_idx - 1)
                else:
                    self.canvas.eraser_idx = max(0, self.canvas.eraser_idx - 1)
        
        self.shutdown()
    
    def shutdown(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("👋 已退出")


def main():
    app = GestureDrawingApp()
    app.run()


if __name__ == "__main__":
    main()