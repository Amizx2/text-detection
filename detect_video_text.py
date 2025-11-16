import argparse
import csv
import os
from pathlib import Path
import cv2
import numpy as np
import easyocr
import time
import itertools
from PIL import Image, ImageDraw, ImageFont

def put_text_utf8_bgr(img_bgr, text, org, font_path, font_size=18, color_bgr=(0,255,0), stroke=1):
    # OpenCV -> PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # шрифт с кириллицей; пример: C:\Windows\Fonts\arial.ttf
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"[Warning] Не удалось загрузить шрифт: {font_path}. Используется стандартный.")
        font = ImageFont.load_default()
    # Рисуем (Pillow ждёт RGB)
    x, y = org
    draw.text((x, y), text, font=font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]),
              stroke_width=stroke, stroke_fill=(0,0,0))
    # PIL -> OpenCV
    img_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_polygon(img, poly, color=(0, 255, 0), thickness=2):
    """poly: список из 4/н точек [[x1,y1],...]."""
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def ensure_dir(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def inpaint_regions(img_bgr, polys, method='telea', radius=3):
    """
    Черновик: удаление служебных оверлеев (инпейнтинг).
    polys: список многоугольников (координаты в пикселях).
    """
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for poly in polys:
        cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    if method == 'telea':
        res = cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_TELEA)
    else:
        res = cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_NS)
    return res

def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return [x1, y1, x2, y2]

def bbox_to_poly(b):
    x1, y1, x2, y2 = b
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = float(areaA + areaB - interArea + 1e-9)
    return interArea / union if union > 0 else 0.0

# =========================
# ПРОСТОЙ ТРЕКЕР ПО IOU
# =========================

class Track:
    _id_iter = itertools.count(1)
    def __init__(self, bbox, text, conf, ttl, ema=0.5):
        self.id = next(Track._id_iter)
        self.bbox = bbox[:]  # [x1,y1,x2,y2]
        self.text = text
        self.conf = conf
        self.ttl = ttl
        self.max_ttl = ttl
        self.ema = ema  # сглаживание координат
    def update(self, bbox, text, conf):
        # EMA сглаживание
        for i in range(4):
            self.bbox[i] = int(self.ema * bbox[i] + (1 - self.ema) * self.bbox[i])
        # текст/уверенность обновляем, если стало лучше
        if conf >= self.conf:
            self.text = text
            self.conf = conf
        self.ttl = self.max_ttl

def greedy_match(tracks, dets, iou_thr):
    """
    tracks: list[Track]
    dets: list of dicts {bbox,text,conf,poly}
    Возвращает кортеж (matches, unmatched_tracks_idx, unmatched_dets_idx),
    где matches = list of (t_idx, d_idx).
    """
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    iou_mat = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    for ti, t in enumerate(tracks):
        for di, d in enumerate(dets):
            iou_mat[ti, di] = iou(t.bbox, d["bbox"])

    matches = []
    used_t = set()
    used_d = set()

    # Жадно берём пары с максимальными IOU
    while True:
        ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[ti, di] < iou_thr:
            break
        if ti in used_t or di in used_d:
            iou_mat[ti, di] = -1
            continue
        matches.append((ti, di))
        used_t.add(ti); used_d.add(di)
        iou_mat[ti, :] = -1
        iou_mat[:, di] = -1

    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_dets = [i for i in range(len(dets)) if i not in used_d]
    return matches, unmatched_tracks, unmatched_dets

# =========================
# ОСНОВНОЙ СКРИПТ
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Детектирование текста на видео (EasyOCR + OpenCV) с памятью и трекингом."
    )
    parser.add_argument("--video_in", required=True, help="Путь к входному видео")
    parser.add_argument("--video_out", default="output/annotated.mp4",
                        help="Путь к выходному видео с разметкой")
    parser.add_argument("--csv_out", default="output/detections.csv",
                        help="CSV-лог (время, bbox/track, текст)")
    parser.add_argument("--langs", nargs="+", default=["en", "ru"],
                        help="Языки для EasyOCR (например: en ru)")
    parser.add_argument("--gpu", action="store_true", help="Использовать GPU, если доступно")
    parser.add_argument("--every_n", type=int, default=1,
                        help="Обрабатывать каждый n-й кадр (для ускорения)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Мин. порог уверенности EasyOCR для принятия бокса")
    parser.add_argument("--draw_text", action="store_true",
                        help="Подписывать распознанный текст над боксом")
    parser.add_argument("--write_csv", action="store_true",
                        help="Сохранить детекции в CSV")
    parser.add_argument("--inpaint", action="store_true",
                        help="Экспериментальная очистка оверлеев инпейнтингом (сохраняется в *_clean.mp4)")
    # Новые флаги
    parser.add_argument("--smooth_hold", type=int, default=5,
                        help="Сколько кадров держать боксы, если распознавание пропало (вариант 2)")
    parser.add_argument("--use_tracker", action="store_true",
                        help="Включить простой IOU-трекер (вариант 3)")
    parser.add_argument("--iou", type=float, default=0.3,
                        help="IOU-порог для матчингa трекера")
    parser.add_argument("--ema", type=float, default=0.5,
                        help="EMA сглаживание координат трекера (0..1, больше — быстрее)")
    # >>> добавили параметры шрифта
    parser.add_argument("--font_path", default=r"C:\Windows\Fonts\arial.ttf",
                        help="Путь к TTF/OTF шрифту для кириллицы (пример: C:\\Windows\\Fonts\\arial.ttf)")
    parser.add_argument("--font_size", type=int, default=18,
                        help="Размер шрифта для подписей")
    parser.add_argument("--font_stroke", type=int, default=2,
                        help="Толщина обводки текста")

    args = parser.parse_args()

    # Инициализация EasyOCR
    t0 = time.time()
    reader = easyocr.Reader(args.langs, gpu=args.gpu)
    print(f"[Init] EasyOCR готов за {time.time() - t0:.2f} c")

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Выходное видео с разметкой
    ensure_dir(args.video_out)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.video_out, fourcc, fps, (w, h))

    # При включенном инпейнтинге сохраняем ещё чистую версию
    writer_clean = None
    if args.inpaint:
        out_clean = str(Path(args.video_out).with_name(
            Path(args.video_out).stem + "_clean.mp4"))
        writer_clean = cv2.VideoWriter(out_clean, fourcc, fps, (w, h))
        print(f"[Info] Чистое видео (инпейнтинг): {out_clean}")

    # CSV-лог
    if args.write_csv:
        ensure_dir(args.csv_out)
        csv_file = open(args.csv_out, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file, delimiter=";")
        csv_writer.writerow([
            "frame_idx", "time_sec",
            "track_id",
            "bbox_x1y1x2y2",
            "conf",
            "text"
        ])
    else:
        csv_file = None
        csv_writer = None

    frame_idx = 0
    memory_results, tracks = [], []

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # будем рисовать на каждом кадре
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            should_run_ocr = (frame_idx % args.every_n == 0)
            detections, polys = [], []

            if should_run_ocr:
                # === OCR только на нужных кадрах ===
                ocr_results = reader.readtext(frame_rgb, detail=1, paragraph=False)
                for (poly, text, conf) in ocr_results:
                    if conf < args.conf:
                        continue
                    bbox = poly_to_bbox(poly)
                    detections.append({"bbox": bbox, "text": text, "conf": conf, "poly": poly})
                    polys.append(poly)

                drawn_items = []
                if args.use_tracker:
                    # трекер обновляем ТОЛЬКО когда был OCR
                    matches, un_tr, un_det = greedy_match(tracks, detections, args.iou)
                    for ti, di in matches:
                        tracks[ti].update(detections[di]["bbox"], detections[di]["text"], detections[di]["conf"])
                    for ti in un_tr:
                        tracks[ti].ttl -= 1
                    for di in un_det:
                        d = detections[di]
                        tracks.append(Track(d["bbox"], d["text"], d["conf"], ttl=args.smooth_hold, ema=args.ema))
                    tracks = [t for t in tracks if t.ttl > 0]
                    for t in tracks:
                        drawn_items.append({"track_id": t.id, "bbox": t.bbox[:], "text": t.text, "conf": t.conf})
                else:
                    # память обновляем ТОЛЬКО когда был OCR
                    if detections:
                        memory_results = [
                            {"bbox": d["bbox"], "text": d["text"], "conf": d["conf"], "ttl": args.smooth_hold}
                            for d in detections]
                    else:
                        # OCR был, но ничего не нашли — уменьшим ttl
                        for m in memory_results:
                            m["ttl"] -= 1
                        memory_results = [m for m in memory_results if m["ttl"] > 0]

                    drawn_items = []
                    for idx, m in enumerate(memory_results, start=1):
                        drawn_items.append({"track_id": idx, "bbox": m["bbox"], "text": m["text"], "conf": m["conf"]})

            else:
                # === OCR НЕ запускаем: просто берём прошлое состояние ===
                drawn_items = []
                if args.use_tracker:
                    # треки НЕ старим на «пропущенных» кадрах, просто рисуем как есть
                    for t in tracks:
                        drawn_items.append({"track_id": t.id, "bbox": t.bbox[:], "text": t.text, "conf": t.conf})
                else:
                    # память тоже НЕ старим между OCR-кадрами
                    for idx, m in enumerate(memory_results, start=1):
                        drawn_items.append({"track_id": idx, "bbox": m["bbox"], "text": m["text"], "conf": m["conf"]})

            # ======== РИСОВАНИЕ И ЛОГИ ========
            for item in drawn_items:
                b = item["bbox"]
                draw_polygon(frame_bgr, bbox_to_poly(b))
                if args.draw_text:
                    label = f"ID {item['track_id']}: {item['text']} ({item['conf']:.2f})"
                    # >>> заменили cv2.putText на Pillow-версию с кириллицей
                    put_text_utf8_bgr(
                        frame_bgr,
                        label,
                        (b[0], max(0, b[1] - 20)),
                        font_path=args.font_path,
                        font_size=args.font_size,
                        color_bgr=(0, 255, 0),
                        stroke=args.font_stroke
                    )
                if csv_writer is not None and should_run_ocr:
                    # логично писать CSV только когда OCR реально делался
                    csv_writer.writerow([frame_idx, f"{frame_idx / fps:.3f}", item["track_id"],
                                         f"{b[0]},{b[1]},{b[2]},{b[3]}",
                                         f"{item['conf']:.3f}", item["text"]])

            # Пишем кадр(ы)
            writer.write(frame_bgr)
            if writer_clean:
                # инпейнт считаем только в OCR-кадры: иначе лишний CPU
                if should_run_ocr and polys:
                    cleaned = inpaint_regions(frame_bgr, polys)
                    writer_clean.write(cleaned)
                else:
                    writer_clean.write(frame_bgr)

            frame_idx += 1
            if frame_idx % max(int(fps), 1) == 0:
                print(f"[Progress] {frame_idx}/{total_frames}")

    finally:
        cap.release()
        writer.release()
        if writer_clean is not None:
            writer_clean.release()
        if csv_file is not None:
            csv_file.close()

    print("[Done] Готово:")
    print(f" - Аннотированное видео: {args.video_out}")
    if writer_clean is not None:
        print(f" - «Чистое» видео без оверлеев: {Path(args.video_out).with_name(Path(args.video_out).stem + '_clean.mp4')}")
    if csv_writer is not None:
        print(f" - CSV-лог: {args.csv_out}")

if __name__ == "__main__":
    main()
