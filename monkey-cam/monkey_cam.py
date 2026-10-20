#!/usr/bin/env python3
"""Monkey Cam: смотрит в настоящую камеру и отдаёт в Zoom/Телемост виртуальную
камеру с обезьяной, которая повторяет мимику и жесты."""
import argparse
import os
import platform
import shutil
import sys
import threading
import time
import types
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = platform.system() == 'Windows'

MOODS = [
    ('default', 'Просто monke'),
    ('cup', 'Со стаканом'),
    ('sad', 'Грустный'),
    ('smile', 'Улыбается'),
    ('think', 'Задумался'),
    ('glasses', 'В очках'),
]
MOOD_NAMES = dict(MOODS)

MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models/'
MODELS = {
    'face_landmarker.task': 'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'hand_landmarker.task': 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'efficientdet_lite0.tflite': 'object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite',
}


def ensure_models(folder):
    os.makedirs(folder, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(folder, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f'Скачиваю модель {name}…', flush=True)
        urllib.request.urlretrieve(MODEL_BASE + url, path + '.part')
        os.replace(path + '.part', path)


def read_bytes(path):
    # Модели и картинки читаем байтами: так работают пути с кириллицей на Windows.
    with open(path, 'rb') as f:
        return f.read()


def load_image(path):
    img = cv2.imdecode(np.frombuffer(read_bytes(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'не получилось открыть картинку {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def quiet_native_logs():
    """MediaPipe пишет служебные логи прямо в stderr и ломает строку статуса.
    Глушим stderr на уровне ОС, а сообщения Python оставляем видимыми."""
    try:
        sys.stderr.flush()
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        sys.stderr = open(saved, 'w', buffering=1, encoding='utf-8', errors='replace')
    except OSError:
        pass


# --- Распознавание: тот же алгоритм, что в веб-версии Monke Mirror ---

def dist(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def eye_darkness(face, rgb):
    h, w = rgb.shape[:2]

    def lum(x, y):
        px = rgb[min(h - 1, max(0, round(y * h))), min(w - 1, max(0, round(x * w)))]
        return px[0] * .2126 + px[1] * .7152 + px[2] * .0722

    skin = (lum(face[50].x, face[50].y) + lum(face[280].x, face[280].y) + lum(face[1].x, face[1].y)) / 3

    def score(a, b):
        l, r = face[a], face[b]
        ew = abs(r.x - l.x)
        dark = 0
        for ix in range(9):
            for iy in range(5):
                x = min(l.x, r.x) + ew * (ix / 8)
                y = (l.y + r.y) / 2 + ew * (iy / 4 - .5) * .65
                if lum(x, y) < min(80, skin * .59):
                    dark += 1
        return dark / 45

    return min(score(33, 133), score(362, 263))


def head_tilt(face):
    return (face[1].y - face[168].y) / max(.02, face[152].y - face[10].y)


def decide(r, baseline):
    face, blend = r['face'], r['blend']
    if not face:
        return 'default'
    b = lambda k: blend.get(k, 0.0)
    fw = abs(face[454].x - face[234].x) or .2
    mouth = types.SimpleNamespace(x=(face[13].x + face[14].x) / 2, y=(face[13].y + face[14].y) / 2)
    up = (b('eyeLookUpLeft') + b('eyeLookUpRight')) / 2
    if up > .20 and any(len(h) > 8 and dist(h[8], mouth) < fw * .23 for h in r['hands']):
        return 'think'
    w, h = r['width'], r['height']
    for o in r['objects']:
        bb = o.bounding_box
        for hand in r['hands']:
            for i in (0, 4, 8, 9, 12):
                p = hand[i]
                if (bb.origin_x - bb.width * .35 < p.x * w < bb.origin_x + bb.width * 1.35 and
                        bb.origin_y - bb.height * .35 < p.y * h < bb.origin_y + bb.height * 1.35):
                    return 'cup'
    blink = (b('eyeBlinkLeft') + b('eyeBlinkRight')) / 2
    if r['eye_dark'] > max(.54, baseline.get('eye_dark', 0) + .22) and blink < .65:
        return 'glasses'
    smile = (b('mouthSmileLeft') + b('mouthSmileRight')) / 2
    if smile > max(.34, baseline.get('smile', 0) + .16):
        return 'smile'
    frown = (b('mouthFrownLeft') + b('mouthFrownRight')) / 2
    if (frown > .25 or (b('browInnerUp') > .5 and smile < .12) or
            head_tilt(face) > baseline.get('nose', .21) + .12):
        return 'sad'
    return 'default'


def make_baseline(r):
    blend = r['blend']
    return {
        'smile': (blend.get('mouthSmileLeft', 0) + blend.get('mouthSmileRight', 0)) / 2,
        'eye_dark': r['eye_dark'],
        'nose': head_tilt(r['face']),
    }


class Stabilizer:
    """Не даёт картинке дёргаться: новая реакция должна продержаться немного."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.current = self.candidate = 'default'
        self.since = 0.0

    def update(self, mood, now):
        if mood != self.candidate:
            self.candidate, self.since = mood, now
        if now - self.since >= (.7 if mood == 'default' else .33):
            self.current = mood
        return self.current


class Detector:
    def __init__(self, models_dir):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        video = vision.RunningMode.VIDEO
        opts = lambda name: BaseOptions(model_asset_buffer=read_bytes(os.path.join(models_dir, name)))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=opts('face_landmarker.task'), running_mode=video, num_faces=1,
            output_face_blendshapes=True, min_face_detection_confidence=.4, min_tracking_confidence=.4))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=opts('hand_landmarker.task'), running_mode=video, num_hands=2,
            min_hand_detection_confidence=.4, min_hand_presence_confidence=.4))
        self.objects = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=opts('efficientdet_lite0.tflite'), running_mode=video, score_threshold=.28,
            category_allowlist=['cup', 'wine glass'], max_results=4))
        self.frame = 0
        self.cups, self.cups_time = [], 0
        self.last_ts = 0

    def close(self):
        for model in (self.face, self.hands, self.objects):
            model.close()

    def process(self, rgb):
        import mediapipe as mp
        ts = max(self.last_ts + 1, int(time.monotonic() * 1000))
        self.last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        fr = self.face.detect_for_video(image, ts)
        hr = self.hands.detect_for_video(image, ts)
        if self.frame % 3 == 0:
            self.cups, self.cups_time = self.objects.detect_for_video(image, ts).detections, ts
        self.frame += 1
        if ts - self.cups_time > 800:
            self.cups = []
        face = fr.face_landmarks[0] if fr.face_landmarks else None
        blend = {c.category_name: c.score for c in fr.face_blendshapes[0]} if fr.face_blendshapes else {}
        return {
            'face': face, 'blend': blend, 'hands': hr.hand_landmarks, 'objects': self.cups,
            'eye_dark': eye_darkness(face, rgb) if face else 0.0,
            'width': rgb.shape[1], 'height': rgb.shape[0],
        }


# --- Картинка для виртуальной камеры ---

def compose(img, width, height):
    """Картинка целиком по центру, по краям размытая копия вместо чёрных полос."""
    ih, iw = img.shape[:2]
    cover = max(width / iw, height / ih)
    bg = cv2.resize(img, (round(iw * cover), round(ih * cover)), interpolation=cv2.INTER_AREA)
    y0, x0 = (bg.shape[0] - height) // 2, (bg.shape[1] - width) // 2
    bg = bg[y0:y0 + height, x0:x0 + width]
    bg = (cv2.GaussianBlur(bg, (0, 0), 28) * .55).astype(np.uint8)
    fit = min(width / iw, height / ih)
    fw, fh = round(iw * fit), round(ih * fit)
    fg = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_CUBIC)
    y, x = (height - fh) // 2, (width - fw) // 2
    bg[y:y + fh, x:x + fw] = fg
    return bg


class Output:
    def __init__(self, images, width, height, fade=.18):
        self.frames = {k: compose(v, width, height) for k, v in images.items()}
        self.fade = fade
        self.shown, self.prev, self.changed = 'default', 'default', -1e9

    def frame(self, mood, now):
        if mood != self.shown:
            self.prev, self.shown, self.changed = self.shown, mood, now
        a = (now - self.changed) / self.fade
        if a >= 1:
            return self.frames[self.shown]
        return cv2.addWeighted(self.frames[self.prev], 1 - a, self.frames[self.shown], a, 0)


def load_images(custom_dir):
    images = {}
    for mood, _ in MOODS:
        path = os.path.join(HERE, 'assets', mood + '.jpg')
        if custom_dir:
            for ext in ('.jpg', '.jpeg', '.png', '.webp'):
                candidate = os.path.join(custom_dir, mood + ext)
                if os.path.exists(candidate):
                    path = candidate
                    break
        images[mood] = load_image(path)
    return images


# --- Камера и клавиатура ---

def open_source(source, width, height):
    if not str(source).isdigit():
        cap = cv2.VideoCapture(source)
    else:
        index = int(source)
        api = cv2.CAP_DSHOW if WINDOWS else cv2.CAP_V4L2 if platform.system() == 'Linux' else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap if cap.isOpened() else None


class Keys:
    """Нажатия в терминале без Enter (Windows и Linux)."""

    def __init__(self):
        self.restore = None
        self.enabled = sys.stdin.isatty()
        if self.enabled and not WINDOWS:
            import termios
            import tty
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self.restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def read(self):
        if not self.enabled:
            return None
        if WINDOWS:
            import msvcrt
            return msvcrt.getwch().lower() if msvcrt.kbhit() else None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None

    def close(self):
        if self.restore:
            self.restore()


class Vision(threading.Thread):
    """Читает камеру и распознаёт мимику в фоне, пока основной цикл кормит виртуальную камеру."""

    def __init__(self, cap, detector, loop_file):
        super().__init__(daemon=True)
        self.cap, self.detector, self.loop_file = cap, detector, loop_file
        self.stable = Stabilizer()
        self.baseline = {}
        self.result = None
        self.mood = 'default'
        self.preview = None
        self.fps = 0.0
        self.error = None
        self.running = True

    def run(self):
        try:
            self.loop()
        finally:
            self.detector.close()
            self.cap.release()

    def loop(self):
        failures, last = 0, time.monotonic()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                failures += 1
                if failures > 50:
                    self.error = 'Камера перестала отдавать кадры. Возможно, её заняло другое приложение.'
                    return
                time.sleep(.05)
                continue
            failures = 0
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                r = self.detector.process(rgb)
            except Exception as e:  # noqa: BLE001 — показываем причину и выходим
                self.error = f'Ошибка распознавания: {e}'
                return
            now = time.monotonic()
            self.result = r
            self.mood = self.stable.update(decide(r, self.baseline), now)
            self.fps = .8 * self.fps + .2 / max(1e-3, now - last)
            last = now
            self.preview = (frame, r)

    def calibrate(self):
        r = self.result
        if not r or not r['face']:
            return False
        self.baseline = make_baseline(r)
        self.stable.reset()
        return True


def draw_preview(frame, r, out_frame, mood, manual):
    img = frame.copy()
    h, w = img.shape[:2]
    for p in r['face'] or []:
        cv2.circle(img, (round(p.x * w), round(p.y * h)), 1, (135, 237, 216), -1)
    for hand in r['hands']:
        for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12), (9, 13, 14, 15, 16),
                      (13, 17, 18, 19, 20), (0, 17)):
            pts = np.array([(round(hand[i].x * w), round(hand[i].y * h)) for i in chain], np.int32)
            cv2.polylines(img, [pts], False, (135, 237, 216), 2)
    img = cv2.flip(img, 1)
    tw = w // 3
    th = round(tw * out_frame.shape[0] / out_frame.shape[1])
    img[h - th - 10:h - 10, w - tw - 10:w - 10] = cv2.cvtColor(cv2.resize(out_frame, (tw, th)), cv2.COLOR_RGB2BGR)
    label = f"{'MANUAL' if manual else 'AUTO'}: {mood}" + ('' if r['face'] else '  (no face)')
    cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (135, 237, 216), 2, cv2.LINE_AA)
    return img


def virtual_camera_help():
    if WINDOWS:
        return ('Не найдена виртуальная камера. Установи OBS Studio с https://obsproject.com/ '
                '(сам OBS запускать не нужно) и запусти Monkey Cam ещё раз. '
                'Если OBS открыт и в нём включена виртуальная камера — выключи её.')
    if platform.system() == 'Linux':
        return ('Не найдена виртуальная камера v4l2loopback. Запусти один раз: sudo ./setup-linux.sh '
                'и затем снова ./run.sh')
    return 'Не найдена виртуальная камера. Установи OBS Studio, один раз включи и выключи в нём виртуальную камеру.'


def main():
    ap = argparse.ArgumentParser(description='Подменяет камеру в Zoom/Телемосте на обезьяну, которая повторяет мимику.')
    ap.add_argument('--camera', default='0', help='номер настоящей камеры (0, 1, …) или путь к видеофайлу; по умолчанию 0')
    ap.add_argument('--size', default='1280x720', help='размер картинки для звонка, по умолчанию 1280x720')
    ap.add_argument('--fps', type=float, default=30)
    ap.add_argument('--backend', default=None, help='obs, unitycapture (Windows) или v4l2loopback (Linux)')
    ap.add_argument('--device', default=None, help='Linux: какое устройство v4l2loopback использовать, например /dev/video10')
    ap.add_argument('--images', default=None, help='папка со своими картинками default/cup/sad/smile/think/glasses.jpg')
    ap.add_argument('--no-preview', action='store_true', help='не показывать окно с камерой')
    ap.add_argument('--no-virtual', action='store_true', help='без виртуальной камеры, только окно проверки')
    ap.add_argument('--debug', action='store_true', help='показывать служебные логи MediaPipe')
    args = ap.parse_args()
    if not args.debug:
        quiet_native_logs()

    try:
        width, height = (int(v) for v in args.size.lower().split('x'))
    except ValueError:
        sys.exit('Размер укажи так: --size 1280x720')

    models_dir = os.path.join(HERE, 'models')
    try:
        ensure_models(models_dir)
    except OSError as e:
        sys.exit(f'Не получилось скачать модели распознавания ({e}). Проверь интернет и запусти ещё раз.')

    images = load_images(args.images)
    output = Output(images, width, height)

    cap = open_source(args.camera, 640, 480)
    if cap is None:
        sys.exit(f'Не получилось открыть камеру {args.camera}. Закрой приложения, которые её используют, '
                 'или попробуй другой номер: --camera 1')

    print('Загружаю распознавание…', flush=True)
    vision = Vision(cap, Detector(models_dir), loop_file=not str(args.camera).isdigit())
    vision.start()

    cam = None
    if not args.no_virtual:
        import pyvirtualcam
        kw = {'device': args.device} if args.device else {}
        try:
            cam = pyvirtualcam.Camera(width, height, args.fps, backend=args.backend, **kw)
        except Exception as e:  # noqa: BLE001 — у бэкендов разные типы ошибок
            vision.running = False
            vision.join(timeout=3)
            print(virtual_camera_help())
            sys.exit(f'Подробности: {e}')

    # Без графического экрана (сервер, SSH) окно OpenCV роняет процесс, поэтому не открываем его.
    has_display = platform.system() != 'Linux' or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    preview = has_display and not args.no_preview
    keys = Keys()
    manual = None
    print()
    if cam:
        print(f'Готово. В Zoom/Телемосте выбери камеру «{cam.device}». Настоящую камеру сейчас держит Monkey Cam.')
        if WINDOWS:
            print('Если этой камеры нет в списке: полностью закрой браузер/Zoom и открой снова '
                  '(после установки OBS надёжнее перезагрузить компьютер).')
    else:
        print('Готово. Виртуальная камера выключена, работает только окно проверки.')
    print('Клавиши: 1–6 — выбрать картинку вручную, A — снова авто, C — калибровка (смотри в камеру с '
          'обычным лицом), P — окно проверки, Q — выход.')
    print('  ' + '  '.join(f'{i + 1} {name}' for i, (_, name) in enumerate(MOODS)))
    print()

    status_at, status_msg = 0.0, ''
    frame_time = 1 / args.fps
    try:
        while True:
            now = time.monotonic()
            if vision.error:
                print()
                sys.exit(vision.error)
            mood = manual or vision.mood
            out = output.frame(mood, now)
            if cam:
                cam.send(out)

            key = keys.read()
            if preview and vision.preview:
                frame, r = vision.preview
                try:
                    cv2.imshow('Monkey Cam', draw_preview(frame, r, out, mood, manual))
                    k = cv2.waitKey(1) & 0xFF
                    key = key or (chr(k).lower() if k < 255 else None)
                    if cv2.getWindowProperty('Monkey Cam', cv2.WND_PROP_VISIBLE) < 1:
                        preview = False
                except cv2.error:
                    preview = False
                    status_msg = 'окно проверки недоступно, работаю без него'
                if not preview:
                    cv2.destroyAllWindows()

            if key in ('q', '\x1b', 'й'):
                break
            if key and key in '123456':
                manual = MOODS[int(key) - 1][0]
                status_msg = 'ручной режим'
            elif key in ('a', 'ф', '0'):
                manual = None
                vision.stable.reset()
                status_msg = 'авто'
            elif key in ('c', 'с'):
                status_msg = 'калибровка сохранена' if vision.calibrate() else 'для калибровки посмотри в камеру'
            elif key in ('p', 'з'):
                preview = has_display and not preview
                if not preview:
                    cv2.destroyAllWindows()

            if now - status_at > .25:
                status_at = now
                r = vision.result
                face = 'нет данных' if r is None else 'лицо есть' if r['face'] else 'лица не видно'
                line = (f"{'РУЧНОЙ' if manual else 'АВТО'} · {MOOD_NAMES[mood]} · {face} · "
                        f"{vision.fps:.0f} к/с" + (f' · {status_msg}' if status_msg else ''))
                cols = shutil.get_terminal_size((80, 20)).columns - 1
                sys.stdout.write('\r' + line[:cols].ljust(cols))
                sys.stdout.flush()

            if cam:
                cam.sleep_until_next_frame()
            else:
                time.sleep(max(0, frame_time - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        vision.running = False
        vision.join(timeout=3)
        if cam:
            cam.close()
        cv2.destroyAllWindows()
        print('\nMonkey Cam выключен.')


if __name__ == '__main__':
    main()
