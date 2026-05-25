import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video/RTSP source")
    parser.add_argument("--model", type=str, default="yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--save", action="store_true", help="Save webcam output video")
    parser.add_argument("--output", type=str, default="outputs/videos/webcam_demo_output.mp4")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam/video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None

    if args.save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print("Saving webcam demo to:", output_path)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            imgsz=args.imgsz,
            classes=[0, 1, 2, 3, 5, 7],
            verbose=False,
        )

        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)

            for box, cls_id, conf, track_id in zip(boxes, class_ids, confs, track_ids):
                x1, y1, x2, y2 = box

                label = f"{COCO_NAMES.get(int(cls_id), cls_id)} ID:{int(track_id)} {conf:.2f}"

                color = (0, 255, 255) if int(cls_id) == 0 else (255, 180, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        cv2.putText(
            frame,
            "Experimental real-time webcam mode",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "ROI-based risk scoring requires camera-specific ROI calibration.",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Crosswalk Risk Warning - Webcam Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
